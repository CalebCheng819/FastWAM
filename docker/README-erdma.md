# FastWAM eRDMA image gate

This image layer is for the four-node, eight-GPU-per-node PAI DLC run. It only
adds Alibaba Cloud's eRDMA userspace provider and diagnostics. PAI supplies the
kernel driver and eRDMA devices when the selected subscription resource type
supports eRDMA.

## Immutable image contract

- Build from an ACR digest, never from a mutable base tag.
- Build and push a new tag; never overwrite an existing tag.
- Use Ubuntu 22.04 (`jammy`), CUDA 12.1 or newer, NCCL 2.19 or newer, and
  `ibverbs-providers=56.2-1.0.3`.
- The formal DLC job must record both the pushed digest and the source commit.
- Do not bake ACR, OSS, or Alibaba Cloud credentials into the image or build
  arguments.

The current source tag used by the older run is:

```text
pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/chengjuntao:cjt-multirobot-benchmark
```

Resolve that tag to a digest in ACR before building. The Dockerfile deliberately
has no default `BASE_IMAGE`, so a build without a digest fails.

## Build and push on a Docker-capable builder

The DSW container does not provide a Docker daemon. Use an ACR Enterprise cloud
build or another controlled builder in `cn-beijing`. For a Docker-capable builder:

```bash
BASE_REF='pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/chengjuntao@sha256:<base-digest>'
DERIVED_TAG='pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/chengjuntao:fastwam-mr-gaudp-erdma-jammy-20260802-r1'
SOURCE_COMMIT='<clean-pushed-fastwam-commit>'

docker login pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com

if docker manifest inspect "${DERIVED_TAG}" >/dev/null 2>&1; then
  echo "Refusing to overwrite existing tag: ${DERIVED_TAG}" >&2
  exit 1
fi

docker build --pull \
  --build-arg "BASE_IMAGE=${BASE_REF}" \
  --build-arg "SOURCE_COMMIT=${SOURCE_COMMIT}" \
  --file docker/Dockerfile.erdma-jammy \
  --tag "${DERIVED_TAG}" \
  .

docker push "${DERIVED_TAG}"
docker image inspect "${DERIVED_TAG}" --format '{{json .RepoDigests}}'
```

Use interactive `docker login`; do not place a password on the command line or
in this repository. If ACR cloud build is used, select this Dockerfile, repository
root as the build context, the new versioned tag above, and the immutable base
digest as the `BASE_IMAGE` build argument.

## Runtime gates before training

PAI's documented DLC defaults use `eth0` for the NCCL socket bootstrap and
`NCCL_IB_HCA=erdma` for the RDMA data path. DSW-to-DSW diagnostics use `eth1`.
Do not copy the DSW socket-interface setting into a DLC job.

The live CPFS mount returned `ENOSPC` when the bundle was fsynced, so the
exclusive, versioned fallback is published at:

```text
/oss-chengjuntao/artifacts/erdma-userspace-56.2-1.0.3
```

Its immutable identities are:

```text
FASTWAM_ERDMA_BUNDLE_SHA256=8f2c1c43d64a7745bea19bfe4cd1383344c9cf32779166f4aa67809ebf1f5fab
FASTWAM_ERDMA_SOURCE_MANIFEST_SHA256=f05443faa27533274ae1b322723e21ac09bd80bd5b2513638dd2619c67552215
COMPLETE_SHA256=275a3885a5ef56e284d29dfe3ad7f21cfa6430bac9dbe2df339b6605d8568240
```

The helper copies those files from the OSS mount to a content-addressed node-local
directory, verifies every SHA again, and atomically publishes the unpacked
provider. Source it in the same non-login shell that starts preflight and
training:

```bash
source docker/prepare-erdma-userspace.sh
fastwam_prepare_erdma_userspace
```

On success it exports `FASTWAM_ERDMA_BUNDLE_SHA256`,
`FASTWAM_ERDMA_SOURCE_MANIFEST_SHA256`, `FASTWAM_ERDMA_ENV_SHA256`,
`FASTWAM_ERDMA_PROVIDER_ROOT`, `FASTWAM_ERDMA_LOCAL_SOURCE_ROOT`,
`IBV_CONFIG_DIR`, `IBV_DRIVERS=erdma`, `RDMAV_DRIVERS=erdma`, `PATH`, and
`LD_LIBRARY_PATH`. The DLC run manifest must record the first three SHA values.

On every DLC node, before starting Accelerate/DeepSpeed:

```bash
fastwam-verify-erdma-userspace dlc
```

Then run FastWAM's existing global all-reduce preflight with the transport gate
enabled:

```bash
export FASTWAM_PREFLIGHT_REQUIRE_ERDMA=1
bash scripts/dlc_preflight.sh 8
```

The job must not continue unless all of the following are present in terminal
evidence:

- `ibv_devinfo` reports every `erdma_*` port as `PORT_ACTIVE`;
- `ERDMA_USERSPACE_GATE=PASS` and `ERDMA_FRAMEWORK_GATE=PASS`;
- all 32 ranks pass the all-reduce and bandwidth gate;
- NCCL logs contain `NET/IB` with `erdma` and do not contain
  `NET/IB : No device found`, `NET/Socket : Using`, or `Using network Socket`.

Run the offline helper tests with:

```bash
bash docker/test-erdma-helpers.sh
```

For a two-node bandwidth diagnostic independent of FastWAM, use PAI's official
`nccl-tests` image and MPIJob procedure. The Docker image includes `ib_write_bw`
for targeted point-to-point diagnosis; the non-invasive overlay may report
`perftest_available=false`. Neither replaces the 32-rank NCCL gate.

Official references:

- <https://help.aliyun.com/en/pai/erdma-distributed-training-with-high-performance-networks>
- <https://help.aliyun.com/en/pai/create-a-training-task>
- <https://help.aliyun.com/zh/pai/create-a-dsw-instance-image>
