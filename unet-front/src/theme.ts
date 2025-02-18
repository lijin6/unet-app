import { extendTheme } from '@chakra-ui/react';

const theme = extendTheme({
  config: {
    initialColorMode: 'light', // 初始主题为浅色模式
    useSystemColorMode: false, // 不使用系统主题
  },
});

export default theme;