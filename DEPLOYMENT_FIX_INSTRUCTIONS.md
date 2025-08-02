# Deployment Fix Instructions

## Summary of Issues Fixed

This document outlines all the issues that were identified and fixed in the AI Trading Agent deployment.

## Issues Identified and Fixed

### 1. Frontend API Connection Issue
**Problem**: Frontend was hardcoded to connect to `http://localhost:5000/api` which failed when deployed on Google Cloud.

**Fix**: Updated `ai-trading-dashboard/src/components/ZerodhaConfig.jsx` to use relative path `/api` instead of absolute localhost URL.

**Files Modified**:
- `ai-trading-dashboard/src/components/ZerodhaConfig.jsx`

### 2. NPM Dependency Conflict
**Problem**: `npm install` was failing due to dependency conflict between `date-fns@4.1.0` and `react-day-picker@8.10.1`.

**Fix**: Added `--legacy-peer-deps` flag to `npm install` command in deployment script.

**Files Modified**:
- `deploy_fixed_v14.sh`

### 3. Docker Build Error
**Problem**: Docker build was failing with "cannot copy to non-directory" error during `COPY . .` command.

**Fix**: Modified `ai-trading-dashboard/Dockerfile` to explicitly copy only necessary files and directories, avoiding conflicts with `node_modules`.

**Files Modified**:
- `ai-trading-dashboard/Dockerfile`

### 4. Backend Init:CrashLoopBackOff (Redis Connection)
**Problem**: Backend pod was failing to start due to Redis connection issues during initialization.

**Fix**: Updated Redis URL configuration in multiple files to use the correct Kubernetes service name and added proper error handling.

**Files Modified**:
- `ai_trading_agent/src/api/zerodha_config.py`
- `ai_trading_agent/src/data_collection/zerodha_client.py`

### 5. Cloud SQL Proxy PodInitializing Error
**Problem**: The `cloud-sql-proxy` sidecar container was stuck in `PodInitializing`, preventing the backend from starting.

**Fix**: Removed the `cloud-sql-proxy` sidecar container from the backend deployment. The backend will now directly connect to the PostgreSQL service within the Kubernetes cluster.

**Files Modified**:
- `kubernetes/backend-deployment-no-cloudsql.yaml` (new file)
- `deploy_fixed_v14.sh` (updated to use the new deployment file)

### 6. InvalidImageName Error
**Problem**: The backend pod showed `InvalidImageName` error because the `YOUR_PROJECT_ID` placeholder in the backend image name was not being replaced.

**Fix**: Updated `deploy_fixed_v14.sh` to use a more robust `sed` command to correctly substitute `YOUR_PROJECT_ID` into `kubernetes/backend-deployment-no-cloudsql.yaml` and added a verification step to print the modified YAML content.

**Files Modified**:
- `deploy_fixed_v14.sh`

## Deployment Steps

1. **Extract the Fixed Code**:
   ```bash
   unzip ai_trading_agent_fixed_v12.zip
   cd ai_trading_agent_final_verified
   ```

2. **Set Execute Permissions**:
   ```bash
   chmod +x deploy_fixed_v14.sh
   ```

3. **Deploy to Google Cloud**:
   ```bash
   ./deploy_fixed_v14.sh YOUR_PROJECT_ID
   ```

4. **Monitor Deployment**:
   ```bash
   kubectl get pods -n ai-trading
   kubectl logs -f deployment/ai-trading-backend -n ai-trading
   ```

## Expected Results

After applying these fixes:

1. **Frontend**: Should successfully connect to the backend API
2. **Backend**: Should start without `Init:CrashLoopBackOff` or `PodInitializing` errors, and without `InvalidImageName` errors.
3. **Zerodha Configuration**: Should be able to save and test API credentials
4. **Overall Application**: Should be fully functional on Google Cloud

## Verification Steps

1. **Check Pod Status**:
   ```bash
   kubectl get pods -n ai-trading
   ```
   All pods should show `Running` status.

2. **Test Frontend Access**:
   Visit the frontend URL and verify the Zerodha configuration page loads.

3. **Test API Connection**:
   Enter valid Zerodha API credentials and test the connection.

## Troubleshooting

If you still encounter issues:

1. **Check Backend Logs**:
   ```bash
   kubectl logs deployment/ai-trading-backend -n ai-trading
   ```

2. **Check Redis Connection**:
   ```bash
   kubectl exec -it deployment/redis-deployment -n ai-trading -- redis-cli ping
   ```

3. **Verify Environment Variables**:
   ```bash
   kubectl describe deployment ai-trading-backend -n ai-trading
   ```

## Key Configuration Changes

### Redis URL Configuration
- **Before**: `redis://localhost:6379/0`
- **After**: `redis://ai-trading-redis-master.ai-trading.svc.cluster.local:6379/0`

### Frontend API Base URL
- **Before**: `http://localhost:5000/api`
- **After**: `/api` (relative path)

### NPM Install Command
- **Before**: `npm install`
- **After**: `npm install --legacy-peer-deps`

### Backend Deployment (Cloud SQL Proxy Removal)
- **Before**: Backend deployment included a `cloud-sql-proxy` sidecar container.
- **After**: `cloud-sql-proxy` sidecar removed. Backend now connects directly to PostgreSQL service.

### Project ID Substitution in Backend Deployment
- **Before**: `YOUR_PROJECT_ID` placeholder in `kubernetes/backend-deployment-no-cloudsql.yaml` was not being replaced.
- **After**: `deploy_fixed_v14.sh` now correctly substitutes `YOUR_PROJECT_ID` into `kubernetes/backend-deployment-no-cloudsql.yaml` and includes a verification step.

## Files in This Package

- `ai_trading_agent_fixed_v12.zip`: Complete fixed codebase
- `DEPLOYMENT_FIX_INSTRUCTIONS.md`: This instruction file
- All original project files with fixes applied

## Support

If you continue to experience issues after applying these fixes, please provide:
1. Pod status output (`kubectl get pods -n ai-trading`)
2. Backend logs (`kubectl logs deployment/ai-trading-backend -n ai-trading`)
3. Any error messages from the browser console

