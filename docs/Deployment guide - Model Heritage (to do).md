# Guida Deployment - Model Heritage MVP

## 🚀 Deployment Locale

### Prerequisiti
```bash
# Verifica versioni
python3 --version  # >= 3.11
node --version      # >= 20.0
pnpm --version      # >= 8.0
```

### Setup Backend
```bash
# Clona e prepara backend
cd model_heritage_backend
source venv/bin/activate

# Installa dipendenze aggiuntive se necessario
pip install flask-cors

# Verifica database
python -c "from src.main import app; app.app_context().push(); from src.models.user import db; db.create_all(); print('Database OK')"

# Avvia server
python run_server.py
```

### Setup Frontend
```bash
# Prepara frontend
cd model_heritage_frontend

# Installa dipendenze
pnpm install

# Configura API URL (se necessario)
echo 'VITE_API_URL=http://localhost:5001' > .env.local

# Avvia dev server
pnpm run dev --host
```

### Verifica Funzionamento
```bash
# Test backend
curl http://localhost:5001/api/models
curl http://localhost:5001/api/stats

# Test frontend
curl http://localhost:5173/
```

## 🐳 Deployment Docker (Futuro)

### Dockerfile Backend
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY run_server.py .

EXPOSE 5001
CMD ["python", "run_server.py"]
```

### Dockerfile Frontend
```dockerfile
FROM node:20-alpine

WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN pnpm install

COPY src/ src/
COPY public/ public/
COPY *.config.js ./

RUN pnpm build
EXPOSE 3000
CMD ["pnpm", "preview", "--host"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  backend:
    build: ./model_heritage_backend
    ports: ["5001:5001"]
    volumes: ["./uploads:/app/uploads"]
    
  frontend:
    build: ./model_heritage_frontend
    ports: ["3000:3000"]
    environment:
      - VITE_API_URL=http://backend:5001
    depends_on: [backend]
```

## ☁️ Deployment Cloud

### Opzione 1: Vercel + Railway
```bash
# Frontend su Vercel
cd model_heritage_frontend
pnpm build
vercel --prod

# Backend su Railway
cd model_heritage_backend
# Aggiungi railway.json
railway deploy
```

### Opzione 2: AWS EC2
```bash
# Setup server
sudo apt update
sudo apt install python3 python3-pip nodejs npm
npm install -g pnpm

# Clone e setup
git clone <repo>
cd model_heritage_backend && python run_server.py &
cd model_heritage_frontend && pnpm run dev --host &

# Nginx reverse proxy
sudo apt install nginx
# Configura proxy per porte 5001/5173
```

### Opzione 3: Heroku
```bash
# Backend
cd model_heritage_backend
echo "web: python run_server.py" > Procfile
heroku create model-heritage-api
git push heroku main

# Frontend  
cd model_heritage_frontend
echo '{"scripts":{"start":"pnpm preview"}}' > package.json
heroku create model-heritage-ui
git push heroku main
```

## 🔧 Configurazione Produzione

### Variabili Ambiente
```bash
# Backend
export FLASK_ENV=production
export DATABASE_URL=postgresql://...
export SECRET_KEY=<random-key>
export UPLOAD_FOLDER=/var/uploads
export MAX_FILE_SIZE=5368709120  # 5GB

# Frontend
export VITE_API_URL=https://api.modelheritage.com
export VITE_ENVIRONMENT=production
```

### Database Produzione
```python
# Migrazione da SQLite a PostgreSQL
pip install psycopg2-binary

# src/main.py
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
```

### Storage Produzione
```python
# Integrazione S3/MinIO per file upload
pip install boto3

# src/routes/models.py
import boto3
s3 = boto3.client('s3')
s3.upload_file(file_path, bucket, key)
```

## 🔒 Sicurezza

### HTTPS
```nginx
# nginx.conf
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location /api/ {
        proxy_pass http://localhost:5001;
    }
    
    location / {
        proxy_pass http://localhost:5173;
    }
}
```

### Rate Limiting
```python
# Backend
from flask_limiter import Limiter
limiter = Limiter(app, key_func=get_remote_address)

@limiter.limit("10 per minute")
@app.route('/api/models', methods=['POST'])
def upload_model():
    pass
```

### Validazione File
```python
# Controlli sicurezza upload
def validate_file(file):
    # Controllo dimensioni
    if file.content_length > MAX_FILE_SIZE:
        raise ValueError("File troppo grande")
    
    # Controllo magic bytes
    magic = file.read(8)
    if not is_valid_model_file(magic):
        raise ValueError("Formato non valido")
    
    # Scan antivirus (opzionale)
    if not virus_scan(file):
        raise ValueError("File sospetto")
```

## 📊 Monitoring

### Logging
```python
# Backend logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Metriche
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram
upload_counter = Counter('model_uploads_total')
processing_time = Histogram('model_processing_seconds')
```

### Health Checks
```python
# src/routes/health.py
@app.route('/health')
def health():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'database': check_database(),
        'storage': check_storage()
    }
```

## 🔄 CI/CD

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Test Backend
        run: |
          cd model_heritage_backend
          python -m pytest
      - name: Test Frontend  
        run: |
          cd model_heritage_frontend
          pnpm test
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Production
        run: |
          # Deploy script
```

## 🚨 Troubleshooting

### Problemi Comuni

#### Backend non si avvia
```bash
# Verifica dipendenze
pip list | grep flask

# Controlla porte
lsof -i :5001

# Log errori
python run_server.py 2>&1 | tee error.log
```

#### Frontend non carica
```bash
# Verifica build
pnpm build

# Controlla API URL
grep -r "localhost:5001" src/

# Network issues
curl -v http://localhost:5001/api/models
```

#### Upload fallisce
```bash
# Verifica permessi directory
ls -la uploads/

# Controlla dimensioni file
du -h test_model.pt

# Test API diretta
curl -X POST http://localhost:5001/api/models \
  -F "file=@test.pt" -v
```

### Performance Issues
```bash
# Monitor risorse
htop
df -h
free -m

# Database performance
sqlite3 src/database/app.db ".schema"
sqlite3 src/database/app.db "EXPLAIN QUERY PLAN SELECT * FROM models"

# Network latency
ping localhost
curl -w "@curl-format.txt" http://localhost:5001/api/models
```

## 📈 Scaling

### Horizontal Scaling
- Load balancer (nginx/HAProxy)
- Multiple backend instances
- Shared database (PostgreSQL)
- Distributed storage (S3/MinIO)

### Vertical Scaling
- Aumentare RAM/CPU server
- SSD storage per database
- CDN per assets statici
- Database connection pooling

### Caching
- Redis per session/cache
- Browser caching headers
- Database query caching
- File upload caching

