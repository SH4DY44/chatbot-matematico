import os
import sys
import json
from pathlib import Path

def check_files():
    """Verificar que todos los archivos necesarios existan"""
    required_files = [
        'app.py',
        'chatbot.py', 
        'index.html',
        'script.js',
        'style.css',
        'requirements.txt',
        'Procfile',
        '.env.example',
        '.gitignore'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Archivos faltantes: {', '.join(missing_files)}")
        return False
    
    print("✅ Todos los archivos necesarios están presentes")
    return True

def check_env_configuration():
    """Verificar configuración de variables de entorno"""
    print("\n🔍 Verificando configuración de entorno...")
    
    # Verificar que .env.example existe y tiene las variables necesarias
    if Path('.env.example').exists():
        with open('.env.example', 'r') as f:
            env_example = f.read()
            
        required_vars = ['GEMINI_API_KEY', 'FLASK_ENV', 'FLASK_DEBUG']
        missing_vars = [var for var in required_vars if var not in env_example]
        
        if missing_vars:
            print(f"❌ Variables faltantes en .env.example: {', '.join(missing_vars)}")
            return False
        
        print("✅ .env.example contiene todas las variables necesarias")
    
    # Verificar que no hay API keys hardcodeadas
    suspicious_files = ['app.py', 'chatbot.py', 'script.js']
    hardcoded_secrets = []
    
    for file in suspicious_files:
        if Path(file).exists():
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Buscar patrones de API keys hardcodeadas
            patterns = [
                'AIza',  # Google API keys empiezan con AIza
                'sk-',   # OpenAI API keys
                'key=',  # Patrones genéricos
            ]
            
            for pattern in patterns:
                if pattern in content and 'os.getenv' not in content.split(pattern)[0][-50:]:
                    # Si encuentra el patrón pero no está cerca de os.getenv
                    if pattern == 'AIza' and len([line for line in content.split('\n') if 'AIza' in line and 'os.getenv' not in line]) > 0:
                        hardcoded_secrets.append(f"{file}: posible API key hardcodeada")
    
    if hardcoded_secrets:
        print(f"⚠️ Posibles secretos hardcodeados encontrados:")
        for secret in hardcoded_secrets:
            print(f"   - {secret}")
        return False
    
    print("✅ No se encontraron API keys hardcodeadas")
    return True

def check_dependencies():
    """Verificar dependencies en requirements.txt"""
    print("\n📦 Verificando dependencias...")
    
    if not Path('requirements.txt').exists():
        print("❌ requirements.txt no encontrado")
        return False
    
    with open('requirements.txt', 'r') as f:
        requirements = f.read().strip().split('\n')
    
    required_packages = ['Flask', 'Flask-CORS', 'requests', 'python-dotenv', 'gunicorn']
    missing_packages = []
    
    for package in required_packages:
        if not any(req.startswith(package) for req in requirements):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Paquetes faltantes: {', '.join(missing_packages)}")
        return False
    
    print("✅ Todas las dependencias necesarias están presentes")
    return True

def check_procfile():
    """Verificar Procfile para Railway"""
    print("\n🚀 Verificando Procfile...")
    
    if not Path('Procfile').exists():
        print("❌ Procfile no encontrado")
        return False
    
    with open('Procfile', 'r') as f:
        procfile_content = f.read().strip()
    
    if 'gunicorn' not in procfile_content:
        print("❌ Procfile no usa gunicorn")
        return False
    
    if 'app:app' not in procfile_content:
        print("❌ Procfile no apunta a app:app")
        return False
    
    print("✅ Procfile configurado correctamente")
    return True

def check_security():
    """Verificar aspectos de seguridad"""
    print("\n🔒 Verificando seguridad...")
    
    # Verificar que .env está en .gitignore
    if Path('.gitignore').exists():
        with open('.gitignore', 'r') as f:
            gitignore = f.read()
        
        if '.env' not in gitignore:
            print("⚠️ .env no está en .gitignore")
            return False
    
    # Verificar que no hay credenciales en los archivos
    sensitive_patterns = ['password', 'secret', 'token', 'credential']
    for file in ['app.py', 'chatbot.py']:
        if Path(file).exists():
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            for pattern in sensitive_patterns:
                if pattern in content and 'os.getenv' not in content:
                    # Verificación más específica para evitar falsos positivos
                    lines_with_pattern = [line for line in content.split('\n') if pattern in line]
                    suspicious_lines = [line for line in lines_with_pattern if 'os.getenv' not in line and '#' not in line]
                    
                    if suspicious_lines:
                        print(f"⚠️ Posible información sensible en {file}: {pattern}")
    
    print("✅ Verificaciones de seguridad completadas")
    return True

def main():
    """Función principal"""
    print("🔍 VERIFICACIÓN PRE-DEPLOYMENT - CHATBOT MATEMÁTICO\n")
    
    checks = [
        ("Archivos necesarios", check_files),
        ("Configuración de entorno", check_env_configuration), 
        ("Dependencias", check_dependencies),
        ("Procfile", check_procfile),
        ("Seguridad", check_security)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        try:
            if check_func():
                passed += 1
            else:
                print(f"❌ {name}: FALLÓ")
        except Exception as e:
            print(f"❌ {name}: ERROR - {str(e)}")
    
    print(f"\n📊 RESULTADO: {passed}/{total} verificaciones pasaron")
    
    if passed == total:
        print("✅ ¡TODO LISTO PARA PRODUCCIÓN!")
        print("\n🚀 Pasos siguientes:")
        print("1. Subir código a GitHub")
        print("2. Crear proyecto en Railway") 
        print("3. Configurar GEMINI_API_KEY en Railway")
        print("4. Verificar deployment en /health")
        return True
    else:
        print("❌ CORRIGE LOS PROBLEMAS ANTES DE SUBIR A PRODUCCIÓN")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)