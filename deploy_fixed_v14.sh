#!/bin/bash

# AI Trading Agent GCP Deployment Script (Fixed Version 14)
# This script automates the deployment of the AI Trading Agent to Google Cloud Platform
# Includes comprehensive fixes for StorageClass and robust PVC binding issues, with corrected kubectl patch commands, shell syntax, frontend build path, and improved PVC waiting logic.
# This version also includes aggressive cleanup and ensures frontend image rebuild.

set -e  # Exit on any error



# Configuration
PROJECT_ID=${1:-"your-gcp-project-id"}
CLUSTER_NAME="ai-trading-cluster"
CLUSTER_ZONE="us-central1-a"
NAMESPACE="ai-trading"
SERVICE_ACCOUNT_NAME="ai-trading-sa"

echo -e "${BLUE}=== AI Trading Agent GCP Deployment (Fixed V14) ===${NC}"
echo -e "${BLUE}Project ID: ${PROJECT_ID}${NC}"
echo -e "${BLUE}Cluster: ${CLUSTER_NAME}${NC}"
echo -e "${BLUE}Zone: ${CLUSTER_ZONE}${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    print_status "Prerequisites check passed."
}

# Set up GCP project and authentication
setup_gcp() {
    print_status "Setting up GCP project..."
    
    gcloud config set project $PROJECT_ID
    gcloud auth configure-docker
    
    # Enable required APIs
    print_status "Enabling required GCP APIs..."
    gcloud services enable container.googleapis.com
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable containerregistry.googleapis.com
    gcloud services enable sqladmin.googleapis.com
    gcloud services enable redis.googleapis.com
    gcloud services enable iam.googleapis.com
    gcloud services enable logging.googleapis.com
    gcloud services enable monitoring.googleapis.com
    
    print_status "GCP setup completed."
}

# Create service account for Cloud SQL access
create_service_account() {
    print_status "Creating service account for Cloud SQL access..."
    
    # Create service account if it doesn\\\"t exist
    if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com &> /dev/null; then
        gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
            --display-name="AI Trading Service Account" \
            --description="Service account for AI Trading Agent Cloud SQL access"
    else
        print_warning "Service account $SERVICE_ACCOUNT_NAME already exists."
    fi
    
    # Grant necessary roles
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
        --role="roles/cloudsql.client"
    
    # Create and download service account key
    if [ ! -f "./cloud-sql-key.json" ]; then
        gcloud iam service-accounts keys create ./cloud-sql-key.json \
            --iam-account=$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com
        print_status "Service account key created: cloud-sql-key.json"
    else
        print_warning "Service account key already exists."
    fi
}

# Create GKE cluster
create_cluster() {
    print_status "Creating GKE cluster..."
    
    if gcloud container clusters describe $CLUSTER_NAME --zone=$CLUSTER_ZONE &> /dev/null; then
        print_warning "Cluster $CLUSTER_NAME already exists. Skipping creation."
    else
        gcloud container clusters create $CLUSTER_NAME \
            --zone=$CLUSTER_ZONE \
            --machine-type=e2-standard-4 \
            --num-nodes=3 \
            --enable-autoscaling \
            --min-nodes=2 \
            --max-nodes=10 \
            --enable-autorepair \
            --enable-autoupgrade \
            --disk-size=50GB \
            --disk-type=pd-ssd \
            --enable-ip-alias \
            --network=default \
            --subnetwork=default \
            --addons=HorizontalPodAutoscaling,HttpLoadBalancing \
            --enable-network-policy \
            --workload-pool=$PROJECT_ID.svc.id.goog \
            --logging=SYSTEM,WORKLOAD \
            --monitoring=SYSTEM
        
        print_status "GKE cluster created successfully."
    fi
    
    # Get cluster credentials
    gcloud container clusters get-credentials $CLUSTER_NAME --zone=$CLUSTER_ZONE
}

# Create Cloud SQL PostgreSQL instance
create_database() {
    print_status "Setting up Cloud SQL PostgreSQL..."
    
    INSTANCE_NAME="ai-trading-postgres"
    
    if gcloud sql instances describe $INSTANCE_NAME &> /dev/null; then
        print_warning "Cloud SQL instance $INSTANCE_NAME already exists. Skipping creation."
    else
        gcloud sql instances create $INSTANCE_NAME \
            --database-version=POSTGRES_15 \
            --tier=db-custom-2-4096 \
            --region=us-central1 \
            --storage-type=SSD \
            --storage-size=50GB \
            --storage-auto-increase \
            --maintenance-window-day=SUN \
            --maintenance-window-hour=03 \
            --maintenance-release-channel=production \
            --authorized-networks=0.0.0.0/0
        
        print_status "Cloud SQL PostgreSQL instance created successfully."
    fi
    
    # Create database and user
    if ! gcloud sql databases describe ai_trading_db --instance=$INSTANCE_NAME &> /dev/null; then
        gcloud sql databases create ai_trading_db --instance=$INSTANCE_NAME
    fi
    
    # Generate secure password and save it
    DB_PASSWORD=$(openssl rand -base64 16)
    echo "$DB_PASSWORD" > ./db-password.txt
    
    # Create user if it doesn\\\\\"t exist
    if ! gcloud sql users describe trading_user --instance=$INSTANCE_NAME &> /dev/null; then
        gcloud sql users create trading_user --instance=$INSTANCE_NAME --password=$DB_PASSWORD
    fi
    
    print_status "Database and user setup completed."
}

# Create Redis instance
create_redis() {
    print_status "Setting up Redis instance..."
    
    REDIS_INSTANCE="ai-trading-redis"
    
    if gcloud redis instances describe $REDIS_INSTANCE --region=us-central1 &> /dev/null; then
        print_warning "Redis instance $REDIS_INSTANCE already exists. Skipping creation."
    else
        gcloud redis instances create $REDIS_INSTANCE \
            --size=1 \
            --region=us-central1 \
            --redis-version=redis_7_0 \
            --tier=basic
        
        print_status "Redis instance created successfully."
    fi
}

# Build and push Docker images
build_images() {
    print_status "Building and pushing Docker images..."
    
    # Build backend image using fixed Dockerfile
    print_status "Building backend image with fixed Dockerfile..."
    cp ai_trading_agent/Dockerfile_fixed ai_trading_agent/Dockerfile
    docker build -t gcr.io/$PROJECT_ID/ai-trading-backend:latest ./ai_trading_agent
    docker push gcr.io/$PROJECT_ID/ai-trading-backend:latest
    
    # Build frontend image
    print_status "Building frontend assets..."
    (cd ai-trading-dashboard && npm install --legacy-peer-deps && npm run build)
    print_status "Building frontend image..."
    # Force rebuild of frontend image to include Nginx config changes
    docker build --no-cache -t gcr.io/$PROJECT_ID/ai-trading-frontend:latest ./ai-trading-dashboard
    docker push gcr.io/$PROJECT_ID/ai-trading-frontend:latest
    
    print_status "Docker images built and pushed successfully."
}

# Create Kubernetes secrets
create_k8s_secrets() {
    print_status "Creating Kubernetes secrets..."
    
    # Create namespace first
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Read database password
    if [ -f "./db-password.txt" ]; then
        DB_PASSWORD=$(cat ./db-password.txt)
    else
        print_error "Database password file not found!"
        exit 1
    fi
    
    # Create database password secret
    kubectl create secret generic ai-trading-db-password \
        --from-literal=password="$DB_PASSWORD" \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create Cloud SQL service account key secret
    if [ -f "./cloud-sql-key.json" ]; then
        kubectl create secret generic cloud-sql-key \
            --from-file=key.json=./cloud-sql-key.json \
            --namespace=$NAMESPACE \
            --dry-run=client -o yaml | kubectl apply -f -
    else
        print_error "Cloud SQL key file not found!"
        exit 1
    fi
    
    print_status "Kubernetes secrets created successfully."
}

# Configure StorageClass and handle PVC issues
configure_storage() {
    print_status "Configuring StorageClass and handling PVC issues..."
    
    # Check if standard StorageClass exists and set it as default
    if kubectl get storageclass standard &> /dev/null; then
        print_status "Setting \"standard\" StorageClass as default..."
        # Corrected kubectl patch command for setting default StorageClass
        # Using single quotes for the entire JSON string and double quotes for internal keys/values
        kubectl patch storageclass standard -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
        print_status "StorageClass \"standard\" is now set as default."
    else
        print_warning "StorageClass \"standard\" not found. Checking for other available StorageClasses..."
        kubectl get storageclass
        print_error "Please ensure a proper StorageClass is available in your cluster."
        exit 1
    fi
    
    # Patch StorageClass to Immediate binding mode
    print_status "Patching \"standard\" StorageClass to Immediate binding mode..."
    # Corrected kubectl patch command for setting volumeBindingMode
    # Using single quotes for the entire JSON string and double quotes for internal keys/values
    kubectl patch storageclass standard -p '{"volumeBindingMode":"Immediate"}'
    print_status "StorageClass \"standard\" is now set to Immediate binding mode."

    # Clean up any existing PVCs that might be stuck
    print_status "Cleaning up any existing PVCs..."
    kubectl delete pvc backend-logs-pvc backend-models-pvc postgres-pvc redis-pvc -n $NAMESPACE --ignore-not-found=true
    
    # Wait a moment for cleanup
    sleep 10
    
    print_status "Storage configuration completed."
}

# Deploy to Kubernetes
deploy_to_k8s() {
    print_status "Deploying to Kubernetes..."
    
    # Update image references in deployment files
    sed "s|YOUR_PROJECT_ID|$PROJECT_ID|g" kubernetes/backend-deployment-no-cloudsql.yaml > kubernetes/backend-deployment-no-cloudsql.yaml.tmp && mv kubernetes/backend-deployment-no-cloudsql.yaml.tmp kubernetes/backend-deployment-no-cloudsql.yaml
    print_status "Verifying backend-deployment-no-cloudsql.yaml after PROJECT_ID substitution:"
    cat kubernetes/backend-deployment-no-cloudsql.yaml
    sed "s|YOUR_PROJECT_ID|$PROJECT_ID|g" kubernetes/frontend-deployment-fixed.yaml > kubernetes/frontend-deployment-fixed.yaml.tmp && mv kubernetes/frontend-deployment-fixed.yaml.tmp kubernetes/frontend-deployment-fixed.yaml
    
    # Apply PVCs first
    print_status "Creating PersistentVolumeClaims..."
    kubectl apply -f kubernetes/backend-logs-pvc.yaml
    kubectl apply -f kubernetes/backend-models-pvc.yaml
    kubectl apply -f kubernetes/postgres-pvc.yaml
    kubectl apply -f kubernetes/redis-pvc.yaml
    
    # Wait for PVCs to be bound with more robust error handling
    print_status "Waiting for PVCs to be bound..."
    
    # Function to check PVC status with retries
    check_pvc_status() {
        local pvc_name=$1
        local max_attempts=120 # 120 attempts * 5 seconds = 600 seconds (10 minutes)
        local attempt=1
        
        print_status "Checking status for PVC: $pvc_name"
        
        while [ $attempt -le $max_attempts ]; do
            if kubectl get pvc $pvc_name -n $NAMESPACE -o jsonpath='{.status.phase}' | grep -q "Bound"; then
                print_status "PVC $pvc_name is Bound."
                return 0
            fi
            print_status "Attempt $attempt/$max_attempts: PVC $pvc_name is not yet Bound. Retrying in 5 seconds..."
            sleep 5
            attempt=$((attempt+1))
        done
        
        print_error "PVC $pvc_name did not bind within timeout after $max_attempts attempts."
        kubectl describe pvc $pvc_name -n $NAMESPACE
        return 1
    }
    
    # Check each PVC individually
    check_pvc_status "backend-logs-pvc"
    check_pvc_status "backend-models-pvc"
    check_pvc_status "postgres-pvc"
    check_pvc_status "redis-pvc"
    
    # Show PVC status for debugging
    print_status "Current PVC status:"
    kubectl get pvc -n $NAMESPACE
    
    # Apply Kubernetes configurations in order
    print_status "Applying Kubernetes configurations..."
    kubectl apply -f kubernetes/configmap.yaml
    kubectl apply -f kubernetes/secrets-fixed.yaml
    kubectl apply -f kubernetes/postgres-init-script-configmap.yaml # Apply the init script configmap
    kubectl apply -f kubernetes/postgres-deployment-corrected.yaml # Use the corrected postgres deployment
    kubectl apply -f kubernetes/redis-deployment-fixed.yaml
    
    # Wait for database and Redis to be ready before deploying backend
    print_status "Waiting for PostgreSQL to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/postgres-deployment -n $NAMESPACE || print_warning "PostgreSQL deployment may not be fully ready."
    
    print_status "Waiting for Redis to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/redis-deployment -n $NAMESPACE || print_warning "Redis deployment may not be fully ready."

    # Remove the kubectl exec command for init.sql as it\"s now handled by volume mount
    # print_status \"Applying init.sql to PostgreSQL...\"
    # kubectl exec -i deployment/postgres-deployment -n $NAMESPACE -- psql -U trading_user -d ai_trading_db < ./init.sql
    # print_status \"init.sql applied successfully.\"

    # Deploy backend and frontend
    kubectl apply -f kubernetes/backend-deployment-no-cloudsql.yaml
    kubectl apply -f kubernetes/frontend-deployment-fixed.yaml # Use fixed frontend deployment
}

# Main execution
check_prerequisites
setup_gcp
create_service_account
create_cluster
create_database
create_redis
build_images
create_k8s_secrets
configure_storage
deploy_to_k8s

print_status "Deployment completed successfully!"

# Get external IP of frontend service
print_status "Getting external IP of frontend service..."
FRONTEND_IP="$(kubectl get service ai-trading-frontend-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"

# Loop until IP is available
while [ -z "$FRONTEND_IP" ]; do
    print_status "Waiting for frontend external IP..."
    sleep 10
    FRONTEND_IP="$(kubectl get service ai-trading-frontend-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
done

print_status "Frontend accessible at: http://$FRONTEND_IP"


