FROM nginxinc/nginx-unprivileged:1.27-alpine

COPY --chmod=0444 deploy/nas_export_download/nginx.conf /etc/nginx/conf.d/default.conf
