import React from "react";
import {
  Flex,
  Box,
  VStack,
  Link,
  Text,
  IconButton,
  useDisclosure,
  useColorMode,
  useColorModeValue,
  Button,
} from "@chakra-ui/react";
import { HamburgerIcon, MoonIcon, SunIcon } from "@chakra-ui/icons";
import { Link as RouterLink } from "react-router-dom";

const Sidebar: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  const { colorMode, toggleColorMode } = useColorMode();

  // 动态背景色和文字颜色
  const bgColor = useColorModeValue("gray.100", "gray.800");
  const textColor = useColorModeValue("black", "white");

  return (
    <Flex minH="100vh">
      {/* 侧边栏 */}
      <Box
        w={isOpen ? "250px" : "0"}
        transition="all 0.2s"
        overflow="hidden"
        bg={bgColor}
        boxShadow="md"
      >
        <VStack p={4} spacing={4} align="stretch">
          {/* 标题 */}
          <Text fontSize="xl" fontWeight="bold" color={textColor}>
            乳腺癌医学平台
          </Text>

          {/* 导航链接 */}
          <VStack spacing={2} align="stretch">
            <Link
              as={RouterLink}
              to="/"
              p={2}
              _hover={{ bg: useColorModeValue("gray.200", "gray.700") }}
              borderRadius="md"
              color={textColor}
            >
              科研成果
            </Link>
            <Link
              as={RouterLink}
              to="/segmentation"
              p={2}
              _hover={{ bg: useColorModeValue("gray.200", "gray.700") }}
              borderRadius="md"
              color={textColor}
            >
              图像分割
            </Link>
            {/* 可继续添加其他导航项 */}
          </VStack>

          {/* 主题切换按钮 */}
          <Button
            onClick={toggleColorMode}
            variant="outline"
            colorScheme={colorMode === "light" ? "blue" : "yellow"}
            mt="auto"
          >
            切换主题 {colorMode === "light" ? <MoonIcon ml={2} /> : <SunIcon ml={2} />}
          </Button>
        </VStack>
      </Box>

      {/* 主内容区 */}
      <Box flex="1" p={4}>
        {/* 侧边栏展开/收起按钮 */}
        <IconButton
          aria-label="Toggle Sidebar"
          icon={<HamburgerIcon />}
          onClick={onToggle}
          mb={4}
        />
        {children}
      </Box>
    </Flex>
  );
};

export default Sidebar;