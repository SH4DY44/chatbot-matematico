from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from chatbot import MathChatBot
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
CORS(app)

# Inicializar el chatbot con IA y gráficas
print("🧠 Inicializando ChatBot Matemático con IA y Gráficas...")
bot = MathChatBot()
print("✅ ChatBot completo listo!")

# === RUTAS DEL FRONTEND ===

@app.route('/')
def index():
    """Servir la página principal"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ChatBot Matemático IA</title>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                .error { color: #ff0000; }
            </style>
        </head>
        <body>
            <h1>🧮 ChatBot Matemático con IA</h1>
            <p class="error">Los archivos del frontend no se encontraron.</p>
            <p>API funcionando en <a href="/health">/health</a></p>
            <p>Verifica que index.html esté en la raíz del proyecto.</p>
        </body>
        </html>
        """, 404

@app.route('/style.css')
def css():
    """Servir el archivo CSS"""
    try:
        with open('style.css', 'r', encoding='utf-8') as f:
            content = f.read()
        
        response = app.response_class(
            content,
            mimetype='text/css'
        )
        # Agregar headers de cache para producción
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response
    except FileNotFoundError:
        return "/* CSS no encontrado */", 404

@app.route('/script.js')
def js():
    """Servir el archivo JavaScript"""
    try:
        with open('script.js', 'r', encoding='utf-8') as f:
            content = f.read()
        
        response = app.response_class(
            content,
            mimetype='application/javascript'
        )
        # Agregar headers de cache para producción
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response
    except FileNotFoundError:
        return "/* JavaScript no encontrado */", 404

# === RUTAS DE LA API ===

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint principal del chatbot"""
    try:
        # Verificar que se recibieron datos
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type debe ser application/json',
                'status': 'bad_request'
            }), 400
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No se recibieron datos JSON',
                'status': 'bad_request'
            }), 400
            
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'error': 'El campo "message" es requerido y no puede estar vacío',
                'status': 'bad_request'
            }), 400
        
        # Log de la consulta (solo en desarrollo)
        if os.environ.get('FLASK_ENV') == 'development':
            print(f"🧠 Consulta: {user_message}")
        
        # Obtener respuesta del chatbot
        bot_response = bot.get_response(user_message)
        
        # Procesar respuesta
        if isinstance(bot_response, dict):
            response_type = bot_response.get('type', 'text')
            
            # Log del tipo de respuesta (solo en desarrollo)
            if os.environ.get('FLASK_ENV') == 'development':
                print(f"✅ Respuesta {response_type}")
            
            # Construir respuesta JSON
            json_response = {
                'response': bot_response.get('response', 'Error interno del servidor'),
                'status': 'success',
                'type': response_type
            }
            
            # Agregar datos de gráfica si existen
            if 'chart_data' in bot_response:
                json_response['chart_data'] = bot_response['chart_data']
            
            return jsonify(json_response)
        else:
            # Compatibilidad con respuestas string
            return jsonify({
                'response': str(bot_response),
                'status': 'success',
                'type': 'text'
            })
    
    except Exception as e:
        # Log del error
        print(f"❌ Error en /api/chat: {str(e)}")
        
        return jsonify({
            'error': 'Error interno del servidor',
            'status': 'internal_error',
            'message': 'Hubo un problema procesando tu consulta. Inténtalo de nuevo.'
        }), 500

@app.route('/health')
def health_check():
    """Health check para monitoreo y verificación"""
    try:
        # Verificar que el chatbot esté funcionando
        test_response = bot.get_response("test")
        chatbot_healthy = isinstance(test_response, (dict, str))
        
        return jsonify({
            'status': 'healthy' if chatbot_healthy else 'degraded',
            'chatbot': 'ai_math_specialist_with_charts',
            'version': '4.0',
            'environment': os.environ.get('FLASK_ENV', 'production'),
            'features': {
                'ai_enabled': True,
                'charts_enabled': True,
                'math_calculations': True,
                'conversation': True
            },
            'chatbot_status': 'operational' if chatbot_healthy else 'error'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/stats')
def get_stats():
    """Obtener estadísticas del chatbot"""
    try:
        stats = bot.get_conversation_stats()
        return jsonify({
            'stats': stats,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': 'No se pudieron obtener las estadísticas',
            'status': 'error'
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset_context():
    """Resetear contexto del chatbot"""
    try:
        result = bot.reset_context()
        return jsonify({
            'message': result,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': 'No se pudo resetear el contexto',
            'status': 'error'
        }), 500

# === MANEJO DE ERRORES GLOBAL ===

@app.errorhandler(404)
def not_found(error):
    """Manejo de páginas no encontradas"""
    return jsonify({
        'error': 'Página no encontrada',
        'status': 'not_found',
        'available_endpoints': [
            '/ - Página principal',
            '/health - Estado del sistema',
            '/api/chat - Chat con el bot',
            '/api/stats - Estadísticas',
            '/api/reset - Resetear contexto'
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Manejo de métodos no permitidos"""
    return jsonify({
        'error': 'Método no permitido',
        'status': 'method_not_allowed'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Manejo de errores internos"""
    return jsonify({
        'error': 'Error interno del servidor',
        'status': 'internal_error'
    }), 500

# === CONFIGURACIÓN DE PRODUCCIÓN ===

if __name__ == '__main__':
    # Configuración automática para diferentes plataformas
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Información de inicio
    print(f"🚀 Iniciando ChatBot Matemático en producción")
    print(f"   📡 Host: {host}")
    print(f"   🔌 Puerto: {port}")
    print(f"   🐛 Debug: {debug}")
    print(f"   🌍 Entorno: {os.environ.get('FLASK_ENV', 'production')}")
    print("")
    print("📊 Características disponibles:")
    print("   • 🧠 IA conversacional con Google Gemini")
    print("   • 🧮 Cálculos matemáticos precisos") 
    print("   • 📈 Gráficas interactivas con Chart.js")
    print("   • 🎯 Interfaz web intuitiva con botones")
    print("   • 🔧 API REST completa")
    print("")
    print("🔗 Endpoints disponibles:")
    print(f"   • http://{host}:{port}/ - Interfaz web")
    print(f"   • http://{host}:{port}/health - Estado del sistema")
    print(f"   • http://{host}:{port}/api/chat - API del chatbot")
    print("")
    print("✨ ¡ChatBot listo para el mundo!")
    
    # Iniciar servidor
    app.run(host=host, port=port, debug=debug)