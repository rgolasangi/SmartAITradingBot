# Quick Start Guide - Fixed Deployment V4

## Prerequisites
- Google Cloud SDK installed and authenticated
- kubectl installed
- Docker installed
- Valid GCP project with billing enabled

## Quick Deployment

1. **Make script executable**:
   ```bash
   chmod +x deploy_fixed_v4.sh
   ```

2. **Run deployment** (replace with your project ID):
   ```bash
   ./deploy_fixed_v4.sh your-gcp-project-id
   ```

3. **Monitor progress**:
   ```bash
   kubectl get pods -n ai-trading -w
   ```

## What's Fixed in V4

✅ **Backend Module Error**: Added missing `textblob` dependency  
✅ **Frontend Nginx Error**: Fixed invalid `gzip_proxied` configuration  
✅ **PostgreSQL Init Error**: Corrected data directory mount path  
✅ **PVC Not Found**: Pre-create all PersistentVolumeClaims  
✅ **Resource Ordering**: Proper dependency chain implementation  
✅ **Enhanced Error Handling**: Better timeout and error management  

## Key Changes in V4

- **deploy_fixed_v4.sh**: PVC pre-creation and enhanced error handling
- **requirements.txt**: Added `textblob==0.18.0.post1`
- **nginx.conf**: Fixed `gzip_proxied` directive
- **postgres-deployment-fixed.yaml**: Corrected PGDATA mount path
- **Separate PVC files**: Individual PVC manifests for proper ordering

## Expected Timeline

- **PVCs**: Bind within 5 minutes
- **PostgreSQL**: Ready within 10 minutes  
- **Redis**: Ready within 3 minutes
- **Backend**: Ready within 5 minutes
- **Frontend**: Ready within 2 minutes

## Post-Deployment

1. Update API credentials in Kubernetes secrets
2. Configure domain and SSL certificates
3. Set up monitoring and alerting
4. Test thoroughly before production use

## Troubleshooting

- **Check logs**: `kubectl logs -f deployment/ai-trading-backend -n ai-trading`
- **Pod status**: `kubectl describe pod <pod-name> -n ai-trading`
- **Service status**: `kubectl get svc -n ai-trading`

For detailed information, see `DEPLOYMENT_FIXES.md`.

