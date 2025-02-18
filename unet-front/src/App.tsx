import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ChakraProvider, ColorModeScript } from '@chakra-ui/react';
import theme from './theme'; // 自定义主题文件
import Sidebar from './components/Sidebar';
import HomePage from './pages/HomePage';
import SegmentationPage from './pages/SegmentationPage';

const App: React.FC = () => {
  return (
    <>
      {/* 添加 ColorModeScript 以支持主题切换 */}
      <ColorModeScript initialColorMode={theme.config.initialColorMode} />
      <ChakraProvider theme={theme}>
        <Router>
          <Sidebar>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/segmentation" element={<SegmentationPage />} />
            </Routes>
          </Sidebar>
        </Router>
      </ChakraProvider>
    </>
  );
};

export default App;