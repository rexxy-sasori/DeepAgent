#!/bin/bash
# DeepAgent Docker Build Script
# Builds and optionally pushes the Docker image for cross-platform deployment

set -e

# Configuration
NAMESPACE=${NAMESPACE:-"liuyunxin"}
IMAGE_NAME=${IMAGE_NAME:-"deepagent"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
REGISTRY=${REGISTRY:-"harbor.xa.xshixun.com:7443/hanfeigeng"}
PUSH=${PUSH:-"false"}

# Platform for cross-compilation (ARM -> x86)
PLATFORM=${PLATFORM:-"linux/amd64"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}DeepAgent Docker Build${NC}"
echo -e "${GREEN}======================================${NC}"

# Get full image name
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${YELLOW}Image name: ${FULL_IMAGE_NAME}${NC}"
echo -e "${YELLOW}Platform: ${PLATFORM}${NC}"
echo -e "${YELLOW}Namespace: ${NAMESPACE}${NC}"

# Build the image with buildx for cross-platform
echo -e "${YELLOW}Building Docker image with buildx...${NC}"
docker buildx build --platform ${PLATFORM} \
  --build-arg BASE_IMAGE=${BASE_IMAGE:-harbor.xa.xshixun.com:7443/hanfeigeng/lmsysorg/sglang:kv-cache-logging-dev-otel-0.8} \
  -t ${FULL_IMAGE_NAME} --load .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo -e "${GREEN}Image: ${FULL_IMAGE_NAME}${NC}"
    
    # Show image size
    IMAGE_SIZE=$(docker images ${FULL_IMAGE_NAME} --format "{{.Size}}")
    echo -e "${GREEN}Image size: ${IMAGE_SIZE}${NC}"
else
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi

# Optionally push to registry
if [ "$PUSH" = "true" ]; then
    echo -e "${YELLOW}Pushing to registry...${NC}"
    docker push ${FULL_IMAGE_NAME}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Push successful!${NC}"
    else
        echo -e "${RED}✗ Push failed!${NC}"
        exit 1
    fi
fi

# Print usage instructions
echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}Next Steps:${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "1. Update k8s/deployment.yaml with image: ${FULL_IMAGE_NAME}"
echo -e "2. Create secrets: kubectl apply -f k8s/secrets.yaml"
echo -e "3. Deploy: kubectl apply -f k8s/"
echo -e "\nTo push this image, run:"
echo -e "  REGISTRY=docker.io/yourusername PUSH=true ./build.sh"
