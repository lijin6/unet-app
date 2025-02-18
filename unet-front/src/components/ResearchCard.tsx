import React from 'react';
import { Box, Image, Text, Card, CardBody, CardHeader, Heading, Stack } from '@chakra-ui/react';

interface ResearchCardProps {
  title: string;
  description: string;
  imageUrl: string;
}

const ResearchCard: React.FC<ResearchCardProps> = ({ title, description, imageUrl }) => {
  return (
    <Card maxW="sm" borderRadius="lg" overflow="hidden" boxShadow="lg">
      <CardHeader>
        <Heading size="md">{title}</Heading>
      </CardHeader>
      <CardBody>
        <Box>
          <Image src={imageUrl} alt={title} borderRadius="md" />
        </Box>
        <Stack mt="4">
          <Text>{description}</Text>
        </Stack>
      </CardBody>
    </Card>
  );
};

export default ResearchCard;