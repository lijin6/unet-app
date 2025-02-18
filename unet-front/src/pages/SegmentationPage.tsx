import React from "react";
import {
  Box,
  Heading,
  Text,
  useColorModeValue,
} from "@chakra-ui/react";
import ImageSegmentation from "../components/ImageSegmentation";

const SegmentationPage: React.FC = () => {
  // 动态背景色和文字颜色
  const bgColor = useColorModeValue("white", "gray.800");
  const textColor = useColorModeValue("black", "white");

  return (
    <Box bg={bgColor} color={textColor} p={8}>
      {/* 页面标题 */}
      <Heading as="h1" size="lg" mb={6} textAlign="center">
        乳腺癌医学图像分割
      </Heading>

      {/* 描述文字 */}
      <Text fontSize="md" mb={6} textAlign="center" color={useColorModeValue("gray.600", "gray.400")}>
        上传图像以进行分割，并查看预测掩码和叠加图像。
      </Text>

      {/* 图像分割组件 */}
      <Box>
        <ImageSegmentation />
      </Box>
    </Box>
  );
};

export default SegmentationPage;