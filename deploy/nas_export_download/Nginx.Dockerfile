FROM nginxinc/nginx-unprivileged:1.27-alpine

USER root
COPY deploy/nas_export_download/nginx.conf /tmp/quant-export-nginx.conf
RUN install -o root -g root -m 0444 /tmp/quant-export-nginx.conf /etc/nginx/conf.d/default.conf \
    && rm /tmp/quant-export-nginx.conf \
    && sed -i -E 's/^worker_processes[[:space:]]+auto;/worker_processes 1;/' /etc/nginx/nginx.conf
USER 101
