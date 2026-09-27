# Render check: does the volume plot actually draw?
#
# The unit tests cover the server's bytes and the transform's maths, and neither
# can see the one thing the owner asked for: a picture. A shader that fails to
# compile, a texture that uploads nothing, a camera that never sees the box --
# all of those leave every test green and the page blank, and a blank page is
# exactly what this row exists to prevent.
#
# So this image loads the live page in headless Chromium, drives the component
# and coordinate-system selectors, and fails unless pixels changed. It runs as a
# bounded Job in the cluster's namespace, never natively: a browser is heavy,
# and the VM has no GPU to give it.
#
# Playwright's own image rather than a python:3.12 base, because it already
# carries the browser and its shared libraries; installing Chromium on a slim
# base is a long list of apt packages that changes under you.
FROM mcr.microsoft.com/playwright:v1.49.0-noble

ENV NODE_ENV=production \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

WORKDIR /opt/rendercheck

COPY dockerfiles/rendercheck/package.json /opt/rendercheck/package.json
RUN npm install --omit=dev --no-audit --no-fund

COPY dockerfiles/rendercheck/check.mjs /opt/rendercheck/check.mjs

# No root: the Job runs as a non-root user with a read-only root filesystem, so
# the browser needs a writable /tmp to put its own profile in.
RUN useradd --create-home --uid 10002 rendercheck && mkdir -p /out && chown rendercheck /out
USER 10002

ENTRYPOINT ["node", "/opt/rendercheck/check.mjs"]
