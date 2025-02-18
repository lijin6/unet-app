import React from "react";
import {
  Box,
  Heading,
  SimpleGrid,
  Text,
  useColorModeValue,
} from "@chakra-ui/react";
import ResearchCard from "../components/ResearchCard";

const HomePage: React.FC = () => {
  // 定义科研成果数据
  const researchData = [
    {
      title: "研究结果1",
      description: "损失函数曲线",
      imageUrl: "/images/1.png",
    },
    {
      title: "研究结果2",
      description: "交并比(IoU)",
      imageUrl: "/images/2.png",
    },
    {
      title: "研究结果3",
      description: "原始图像示例一",
      imageUrl: "/images/3.png",
    },
    {
      title: "研究结果4",
      description: "原始图像示例二",
      imageUrl: "/images/4.png",
    },
    { title: "研究结果5", description: "叠加图像", imageUrl: "/images/5.png" },
    { title: "研究结果6", description: "预测掩码", imageUrl: "/images/6.png" },
  ];

  // 动态背景色和文字颜色
  const bgColor = useColorModeValue("white", "gray.800");
  const textColor = useColorModeValue("black", "white");

  return (
    <Box bg={bgColor} color={textColor} p={8}>
      {/* 页面标题 */}
      <Heading as="h1" size="lg" mb={6} textAlign="center">
        科研成果展示
      </Heading>

      {/* 科研成果卡片网格 */}
      <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
        {researchData.map((item, index) => (
          <ResearchCard key={index} {...item} />
        ))}
      </SimpleGrid>
    </Box>
  );
};

export default HomePage;