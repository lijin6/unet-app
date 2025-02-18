import React, { useState } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import {
  Box,
  Image,
  Text,
  Card,
  CardBody,
  CardHeader,
  Heading,
  Stack,
  Flex,
  Button,
  Alert,
  AlertIcon,
  Spinner,
  Center,
  Icon,
} from "@chakra-ui/react";
import { FiUploadCloud } from "react-icons/fi"; // 上传图标

const ImageSegmentation = () => {
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [segmentedResults, setSegmentedResults] = useState<
    { pred_mask: string; overlay: string }[]
  >([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false); // 加载状态

  // 文件上传处理函数
  const onDrop = async (files: File[]) => {
    const file = files[0];
    if (!file.type.startsWith("image/")) {
      setError("仅支持图像文件！");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError("文件过大，请上传小于 10MB 的图片！");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);

    try {
      setError(null);
      setIsLoading(true); // 开始加载

      // 显示原图
      const reader = new FileReader();
      reader.onload = () => setOriginalImage(reader.result as string);
      reader.readAsDataURL(file);

      // 调用后端 API 获取分割结果
      const response = await axios.post(
        "http://localhost:5000/predict", // 使用完整的后端 URL
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      // 检查返回的分割结果是否有效
      if (response.data.pred_mask && response.data.overlay) {
        setSegmentedResults([
          {
            pred_mask: response.data.pred_mask,
            overlay: response.data.overlay,
          },
        ]);
      } else {
        setError("无效的分割结果！");
      }
    } catch (error) {
      console.error("Error:", error.response ? error.response.data : error.message);
      setError("服务器请求失败，请稍后再试！");
    } finally {
      setIsLoading(false); // 结束加载
    }
  };

  // 错误处理函数
  const handleError = (message: string) => {
    setError(message);
    console.error(message);
  };

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  return (
    <Box p={8}>
      {/* 文件上传区域 */}
      <Card maxW="600px" mx="auto" p={6} textAlign="center">
        <CardBody>
          <Heading size="lg" mb={4}>
            图像分割工具
          </Heading>
          <Text color="gray.500" mb={6}>
            拖拽图像至此或点击上传
          </Text>

          {/* 文件上传区域 */}
          <Box
            {...getRootProps()}
            p={8}
            border="2px dashed"
            borderColor="blue.300"
            borderRadius="md"
            cursor="pointer"
            _hover={{ bg: "blue.50" }}
          >
            <input {...getInputProps()} />
            <Icon as={FiUploadCloud} boxSize={8} color="blue.500" mb={2} />
            <Text fontWeight="bold" color="blue.500">
              点击或拖拽上传图片
            </Text>
          </Box>

          {/* 错误提示 */}
          {error && (
            <Alert status="error" mt={4}>
              <AlertIcon />
              {error}
            </Alert>
          )}

          {/* 加载状态 */}
          {isLoading && (
            <Center mt={4}>
              <Spinner size="lg" color="blue.500" />
            </Center>
          )}
        </CardBody>
      </Card>

      {/* 原图和分割结果展示 */}
      {(originalImage || segmentedResults.length > 0) && (
        <Flex mt={8} gap={6} justifyContent="space-between">
          {/* 原图 */}
          <Card flex={1} maxW="45%">
            <CardHeader>
              <Heading size="md">原图</Heading>
            </CardHeader>
            <CardBody>
              {originalImage ? (
                <Image
                  src={originalImage}
                  alt="原图"
                  borderRadius="md"
                  boxShadow="md"
                  border="1px solid"
                  borderColor="gray.200"
                  onError={() => handleError("无法加载原图！")}
                />
              ) : (
                <Text>等待上传图像...</Text>
              )}
            </CardBody>
          </Card>

          {/* 分割结果 */}
          <Card flex={1} maxW="45%">
            <CardHeader>
              <Heading size="md">分割结果</Heading>
            </CardHeader>
            <CardBody>
              {segmentedResults.length > 0 ? (
                <Flex gap={4} justifyContent="center">
                  {/* 预测掩码 */}
                  <Box flex={1} textAlign="left">
                    <Text fontWeight="bold" mb={2}>
                      预测掩码
                    </Text>
                    <Image
                      src={segmentedResults[0].pred_mask}
                      alt="预测掩码"
                      borderRadius="md"
                      boxShadow="md"
                      border="1px solid"
                      borderColor="gray.200"
                      onError={() => handleError("无法加载预测掩码图片！")}
                      justifyContent={'center'}
                    />
                  </Box>

                  {/* 叠加图像 */}
                  <Box flex={1} textAlign="left" justifyContent={'left'}>
                    <Text fontWeight="bold" mb={2} justifyContent={'left'}>
                      叠加图像
                    </Text>
                    <Image
                      src={segmentedResults[0].overlay}
                      alt="叠加图像"
                      borderRadius="md"
                      boxShadow="md"
                      border="1px solid"
                      borderColor="gray.200"
                      onError={() => handleError("无法加载叠加图像！")}
                      
                    />
                  </Box>
                </Flex>
              ) : (
                <Text>等待分割结果...</Text>
              )}
            </CardBody>
          </Card>
        </Flex>
      )}
    </Box>
  );
};

export default ImageSegmentation;