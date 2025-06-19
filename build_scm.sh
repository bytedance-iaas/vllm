#!/bin/bash
set -e

ROOT_DIR=$(pwd)
BUILD_TIME=$(date +%Y%m%d%H%M)
BYTED_MIRROR="mirrors.byted.org"

# 获取 git 分支名
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
# 如果分支是以 release_ 或 release/ 开头，则将 release_ 或 release/ 替换为空
if [[ $BRANCH_NAME =~ ^release[\/_] ]]; then
    echo "release branch"
    BRANCH_NAME=${BRANCH_NAME#release}
    BRANCH_NAME=${BRANCH_NAME#/}
    BRANCH_NAME=${BRANCH_NAME#_}
    # 如果分支里还有 / 或 _ ，则将 / 或 _ 替换为 .
    BRANCH_NAME=${BRANCH_NAME//\//.}
    BRANCH_NAME=${BRANCH_NAME//_/.}
    if [[ ! -z $BRANCH_NAME ]]; then
        BRANCH_NAME=.${BRANCH_NAME}
    fi
    VERSION_SUFFIX=+byted${BRANCH_NAME}.${BUILD_TIME}
elif [[ $BRANCH_NAME == iaas_main ]]; then
    echo "iaas_main branch"
    VERSION_SUFFIX=+iaas.dev.${BUILD_TIME}
else
    echo "not release branch"
    VERSION_SUFFIX=+byted.${BUILD_TIME}
fi

echo "VERSION_SUFFIX: $VERSION_SUFFIX"
VERSION=$(git tag --merged HEAD -l "v[0-9]*.[0-9]*.[0-9]*" | sort -V | tail -n 1 | sed 's/^v//')
if [ -z "$VERSION" ]; then
    echo "no version found, please check the git tag. (DO NOT use git shallow clone)"
    exit 1
fi
WHEEL_VERSION=$VERSION$VERSION_SUFFIX
echo "WHEEL_VERSION: $WHEEL_VERSION"

# 如果是 SCM 构建，则准备 docker 环境
if [[ "${SCM_BUILD}" == "True" ]]; then
    source /root/start_dockerd.sh
fi

# 备份 Dockerfile
cp docker/Dockerfile docker/Dockerfile.bak

# sed -i "s/version\s*=\s*get_vllm_version(),/version=\"$WHEEL_VERSION\",/" setup.py
sed -i 's/FROM nvidia\/cuda/FROM iaas-gpu-cn-beijing.cr.volces.com\/nvcr.io\/nvidia\/cuda/' docker/Dockerfile
sed -i "s/python3 setup.py bdist_wheel/SETUPTOOLS_SCM_PRETEND_VERSION=$WHEEL_VERSION python3 setup.py bdist_wheel/" docker/Dockerfile
sed -i "s@apt-get update@sed -i \"s|http://archive.ubuntu.com|http://$BYTED_MIRROR|g\" /etc/apt/sources.list ; sed -i \"s|http://security.ubuntu.com|http://$BYTED_MIRROR|g\" /etc/apt/sources.list ; apt-get update@" ./docker/Dockerfile
sed -i "s@add-apt-repository@http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= add-apt-repository@" ./docker/Dockerfile

cp -r examples vllm/examples

proxy_args=" --build-arg no_proxy=ppa.launchpad.net,$no_proxy"
if [ ! -z "$http_proxy" ]; then
    proxy_args="$proxy_args --build-arg http_proxy=$http_proxy"
fi
if [ ! -z "$https_proxy" ]; then
    proxy_args="$proxy_args --build-arg https_proxy=$https_proxy"
fi

if [ ! -z "$CUSTOM_CUDA_VERSION" ]; then
    cuda_version_build_arg="--build-arg CUDA_VERSION=$CUSTOM_CUDA_VERSION"
fi

DOCKER_BUILDKIT=1 docker build --network host --no-cache $proxy_args --build-arg max_jobs=256 --build-arg USE_SCCACHE=0 --build-arg GIT_REPO_CHECK=0 $cuda_version_build_arg --build-arg RUN_WHEEL_CHECK=false --tag vllm-ci:build-image --target build --progress plain -f docker/Dockerfile .

# 恢复 Dockerfile
mv docker/Dockerfile.bak docker/Dockerfile

mkdir -p artifacts

docker run --rm -v $(pwd)/artifacts:/artifacts_host vllm-ci:build-image bash -c 'cp -r dist /artifacts_host && chmod -R a+rw /artifacts_host'

# 拷贝到 output 目录
OUTPUT_PATH=$ROOT_DIR/output
rm -rf $OUTPUT_PATH || true
mkdir -p $OUTPUT_PATH
cp artifacts/dist/vllm-*.whl $OUTPUT_PATH/

# 上传 whl 包到 TOS
TOS_UTIL_URL=https://tos-tools.tos-cn-beijing.volces.com/linux/amd64/tosutil
if [ ! -z "$CUSTOM_TOS_UTIL_URL" ]; then
    TOS_UTIL_URL=$CUSTOM_TOS_UTIL_URL
fi
if [ -z "$CUSTOM_TOS_AK" ] && [ -z "$CUSTOM_TOS_SK" ]; then
    echo "CUSTOM_TOS_AK and CUSTOM_TOS_SK are not set, skip uploading to tos"
else
    # 上传制品到 tos
    wget $TOS_UTIL_URL -O tosutil && chmod +x tosutil
    for wheel_file in $(find $OUTPUT_PATH -name "*.whl"); do
        echo "uploading $wheel_file to tos..."
        ./tosutil cp $wheel_file tos://${CUSTOM_TOS_BUCKET}/packages/vllm/$(basename $wheel_file) -re cn-beijing -e tos-cn-beijing.volces.com -i $CUSTOM_TOS_AK -k $CUSTOM_TOS_SK
    done
fi
