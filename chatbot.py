import re
import math
import random
import requests
import json
import os
import time
from decimal import Decimal, getcontext
from fractions import Fraction
import ast
import operator

# Configurar precisión decimal
getcontext().prec = 50

class MathChatBot:
    def __init__(self):
        """Inicializar el chatbot matemático con IA y gráficas"""
        self.context = []
        self.conversation_history = []
        self.last_result = None
        self.variables = {}
        
        # Rate limiting para la IA
        self.last_api_call = 0
        self.min_interval = 1.5  # 1.5 segundos entre llamadas a la IA
        
        # API Key de Google Gemini - SOLO desde variable de entorno
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            print("⚠️ ADVERTENCIA: GEMINI_API_KEY no configurada. IA funcionará en modo fallback.")
            self.ai_available = False
        else:
            self.ai_available = True
            self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        
        # Operadores seguros para evaluación
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        # Funciones matemáticas para cálculo preciso
        self.math_functions = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
            'log': math.log, 'log10': math.log10, 'log2': math.log2,
            'exp': math.exp, 'sqrt': math.sqrt, 'abs': abs,
            'ceil': math.ceil, 'floor': math.floor, 'round': round,
            'factorial': math.factorial, 'gcd': math.gcd,
            'pi': math.pi, 'e': math.e, 'tau': math.tau,
            'degrees': math.degrees, 'radians': math.radians
        }
        
        print("🧠 ChatBot con IA matemática y gráficas inicializado")
        
        # Probar conexión con IA solo si está disponible
        if self.ai_available:
            self.test_ai_connection()
    
    def test_ai_connection(self):
        """Probar si la conexión con Google Gemini funciona"""
        if not self.ai_available:
            return False
            
        try:
            test_response = self.get_ai_response_sync("Responde solo: OK")
            if test_response and "ok" in test_response.lower():
                print("✅ Conexión con IA establecida correctamente")
                return True
            else:
                print("⚠️ Problema con la IA, funcionando en modo fallback")
                self.ai_available = False
                return False
        except Exception as e:
            print(f"⚠️ Error probando IA: {str(e)}")
            self.ai_available = False
            return False
    
    def is_chart_request(self, message):
        """Detectar si el usuario quiere una gráfica"""
        chart_keywords = [
            'grafica', 'gráfica', 'graficar', 'plot', 'dibuja', 'muestra',
            'visualiza', 'compara', 'traza', 'representa'
        ]
        
        function_keywords = [
            'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'abs',
            'x^2', 'x²', 'x^3', 'x³', 'cuadrática', 'cúbica', 'f(x)'
        ]
        
        # Palabras que NO son gráficas (para evitar falsos positivos)
        exclude_keywords = [
            'despejar', 'resolver', 'ecuacion', 'ecuación', 'resultado de',
            'calcular', 'cuanto', 'cuánto', 'valor de'
        ]
        
        # Verificar exclusiones primero
        if any(keyword in message.lower() for keyword in exclude_keywords):
            return False
        
        # Verificar palabras clave de gráficas
        has_chart_keyword = any(keyword in message.lower() for keyword in chart_keywords)
        
        # Verificar si menciona funciones matemáticas para graficar
        has_function = any(func in message.lower() for func in function_keywords)
        
        # Verificar patrones algebraicos (3x+2, 2x-1, etc.)
        algebraic_patterns = [
            r'\d*\.?\d*x\s*[+\-]\s*\d+',  # 3x+2, x-1, etc.
            r'\d*\.?\d*x\^?\d*',          # 3x, 2x^2, x^3, etc.
            r'f\(x\)\s*=',                # f(x) = ...
        ]
        
        has_algebraic = any(re.search(pattern, message.lower()) for pattern in algebraic_patterns)
        
        # Patrones específicos
        chart_patterns = [
            r'grafica?\s+\w+\s*\(',  # "grafica sin(x)"
            r'compara?\s+\w+.*vs.*\w+',  # "compara sin vs cos"
            r'dibuja?\s+la\s+función',  # "dibuja la función"
        ]
        
        has_pattern = any(re.search(pattern, message.lower()) for pattern in chart_patterns)
        
        return has_chart_keyword or has_algebraic or (has_function and any(word in message.lower() for word in ['grafica', 'dibuja', 'muestra', 'compara']))
    
    def normalize_function(self, function_str):
        """Convertir función a formato evaluable por Python"""
        # Reemplazos básicos
        replacements = {
            '^': '**',
            '×': '*',
            '÷': '/',
        }
        
        for old, new in replacements.items():
            function_str = function_str.replace(old, new)
        
        # Agregar * donde sea necesario (3x -> 3*x)
        function_str = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', function_str)
        
        # Agregar * entre paréntesis y variables (2(x+1) -> 2*(x+1))
        function_str = re.sub(r'(\d+)\(', r'\1*(', function_str)
        
        return function_str
    
    def parse_chart_request(self, message):
        """Analizar qué tipo de gráfica quiere el usuario - VERSIÓN MEJORADA"""
        message_lower = message.lower()
        
        # Detectar funciones específicas
        functions = []
        
        # === PARSER PARA FUNCIONES ALGEBRAICAS GENERALES ===
        
        # Buscar patrones de funciones algebraicas (3x+2, 2x-1, etc.)
        algebraic_patterns = [
            r'f\(x\)\s*=\s*([^,\.!?]+)',  # f(x) = 3x+2
            r'grafica?\s+([^,\.!?]+?)(?:\s+de\s+|\s*$)',  # grafica 3x+2
            r'dibuja?\s+([^,\.!?]+?)(?:\s+de\s+|\s*$)',   # dibuja 3x+2
        ]
        
        # Buscar funciones algebraicas
        function_found = False
        for pattern in algebraic_patterns:
            match = re.search(pattern, message_lower)
            if match:
                function_str = match.group(1).strip()
                
                # Verificar si es una función algebraica válida
                if self.is_algebraic_function(function_str):
                    # Convertir a formato evaluable
                    normalized = self.normalize_function(function_str)
                    
                    functions.append({
                        'type': 'algebraic',
                        'expression': normalized,
                        'original': function_str
                    })
                    function_found = True
                    break
        
        # === FUNCIONES PREDEFINIDAS (como antes) ===
        
        if not function_found:
            # Funciones trigonométricas
            if 'sin' in message_lower and 'asin' not in message_lower:
                functions.append({'type': 'predefined', 'name': 'sin'})
            if 'cos' in message_lower and 'acos' not in message_lower:
                functions.append({'type': 'predefined', 'name': 'cos'})
            if 'tan' in message_lower and 'atan' not in message_lower:
                functions.append({'type': 'predefined', 'name': 'tan'})
            
            # Funciones algebraicas simples
            if any(term in message_lower for term in ['x^2', 'x²', 'cuadrática']) and not functions:
                functions.append({'type': 'predefined', 'name': 'x^2'})
            if any(term in message_lower for term in ['x^3', 'x³', 'cúbica']) and not functions:
                functions.append({'type': 'predefined', 'name': 'x^3'})
            
            # Otras funciones
            if 'log' in message_lower:
                functions.append({'type': 'predefined', 'name': 'log'})
            if 'exp' in message_lower:
                functions.append({'type': 'predefined', 'name': 'exp'})
            if 'sqrt' in message_lower or '√' in message:
                functions.append({'type': 'predefined', 'name': 'sqrt'})
            if 'abs' in message_lower:
                functions.append({'type': 'predefined', 'name': 'abs'})
        
        # Detectar rangos
        range_match = re.search(r'de\s+(-?\d+)\s+a\s+(-?\d+)', message_lower)
        if range_match:
            x_range = [int(range_match.group(1)), int(range_match.group(2))]
        else:
            x_range = [-10, 10]  # Rango por defecto
        
        # Detectar comparaciones
        is_comparison = any(word in message_lower for word in ['compara', 'vs', 'versus', 'contra'])
        
        return {
            'functions': functions,
            'range': x_range,
            'is_comparison': is_comparison,
            'type': 'comparison' if is_comparison else 'single'
        }
    
    def is_algebraic_function(self, function_str):
        """Verificar si una cadena es una función algebraica válida"""
        # Patrones para funciones algebraicas válidas
        valid_patterns = [
            r'^[\d\.\s]*x[\d\.\s\+\-\*\/\^\(\)]*$',  # Contiene x y operadores matemáticos
            r'^\d+[\+\-]\d+$',                       # Funciones constantes como 5+2
        ]
        
        # Debe contener 'x' para ser una función
        if 'x' not in function_str:
            return False
        
        # Verificar caracteres permitidos
        allowed_chars = set('0123456789x+-*/^(). ')
        if not all(c in allowed_chars for c in function_str):
            return False
        
        return True
    
    def generate_chart_data(self, chart_info):
        """Generar datos para la gráfica - VERSIÓN MEJORADA"""
        try:
            x_min, x_max = chart_info['range']
            x_values = []
            datasets = []
            
            # Generar valores de x
            x_values = [x_min + i * (x_max - x_min) / 200 for i in range(201)]
            
            # Colores para diferentes funciones
            colors = [
                '#ff0000',  # Rojo
                '#0066cc',  # Azul
                '#009900',  # Verde
                '#ff9900',  # Naranja
                '#990099',  # Morado
                '#cc6600',  # Marrón
            ]
            
            # Generar datos para cada función
            for i, func_info in enumerate(chart_info['functions']):
                y_values = []
                
                # Determinar el tipo de función
                if isinstance(func_info, dict):
                    if func_info['type'] == 'algebraic':
                        # Función algebraica personalizada
                        func_expression = func_info['expression']
                        label = func_info['original']
                        
                        for x in x_values:
                            try:
                                # Reemplazar x en la expresión
                                expression = func_expression.replace('x', f'({x})')
                                y = eval(expression, {"__builtins__": {}, "math": math})
                                
                                # Limitar valores extremos
                                if abs(y) > 1000:
                                    y = None
                                
                                y_values.append(y)
                            except:
                                y_values.append(None)
                        
                    elif func_info['type'] == 'predefined':
                        # Función predefinida
                        func_name = func_info['name']
                        label = func_name + '(x)'
                        
                        # Lógica existente para funciones predefinidas
                        for x in x_values:
                            try:
                                if func_name == 'sin':
                                    y = math.sin(x)
                                elif func_name == 'cos':
                                    y = math.cos(x)
                                elif func_name == 'tan':
                                    y = math.tan(x)
                                    if abs(y) > 10:
                                        y = None
                                elif func_name == 'log':
                                    y = math.log(x) if x > 0 else None
                                elif func_name == 'exp':
                                    y = math.exp(x)
                                    if y > 1000:
                                        y = None
                                elif func_name == 'x^2':
                                    y = x ** 2
                                elif func_name == 'x^3':
                                    y = x ** 3
                                elif func_name == 'sqrt':
                                    y = math.sqrt(x) if x >= 0 else None
                                elif func_name == 'abs':
                                    y = abs(x)
                                else:
                                    y = None
                                
                                y_values.append(y)
                            except:
                                y_values.append(None)
                else:
                    # Compatibilidad con formato anterior (string)
                    func = func_info
                    label = func + '(x)' if func not in ['x^2', 'x^3'] else func
                    
                    # Lógica existente para funciones string
                    for x in x_values:
                        try:
                            if func == 'sin':
                                y = math.sin(x)
                            elif func == 'cos':
                                y = math.cos(x)
                            elif func == 'tan':
                                y = math.tan(x)
                                if abs(y) > 10:
                                    y = None
                            elif func == 'log':
                                y = math.log(x) if x > 0 else None
                            elif func == 'exp':
                                y = math.exp(x)
                                if y > 1000:
                                    y = None
                            elif func == 'x^2':
                                y = x ** 2
                            elif func == 'x^3':
                                y = x ** 3
                            elif func == 'sqrt':
                                y = math.sqrt(x) if x >= 0 else None
                            elif func == 'abs':
                                y = abs(x)
                            else:
                                y = None
                            
                            y_values.append(y)
                        except:
                            y_values.append(None)
                
                # Crear dataset
                dataset = {
                    'label': label,
                    'data': [{'x': x, 'y': y} for x, y in zip(x_values, y_values) if y is not None],
                    'borderColor': colors[i % len(colors)],
                    'backgroundColor': colors[i % len(colors)] + '20',
                    'tension': 0.4,
                    'pointRadius': 0,
                    'fill': False
                }
                
                datasets.append(dataset)
            
            return {
                'type': 'line',
                'data': {
                    'datasets': datasets
                },
                'options': {
                    'responsive': True,
                    'interaction': {
                        'intersect': False,
                        'mode': 'index'
                    },
                    'scales': {
                        'x': {
                            'type': 'linear',
                            'title': {
                                'display': True,
                                'text': 'x'
                            },
                            'grid': {
                                'color': '#e0e0e0'
                            }
                        },
                        'y': {
                            'title': {
                                'display': True,
                                'text': 'y'
                            },
                            'grid': {
                                'color': '#e0e0e0'
                            }
                        }
                    },
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': f"Gráfica de {', '.join([f['original'] if isinstance(f, dict) and 'original' in f else (f['name'] if isinstance(f, dict) else f) for f in chart_info['functions']])}"
                        },
                        'legend': {
                            'display': len(datasets) > 1
                        }
                    }
                }
            }
            
        except Exception as e:
            print(f"Error generando datos de gráfica: {str(e)}")
            return None
    
    def is_mathematical_expression(self, message):
        """Detectar si es una expresión matemática que se puede calcular directamente"""
        # No tratar como expresión matemática si es una solicitud de gráfica
        if self.is_chart_request(message):
            return False
            
        # Limpiar el mensaje
        cleaned = message.strip().lower()
        
        # Patrones que indican cálculo matemático directo
        math_patterns = [
            r'\d+\s*[+\-*/^%]\s*\d+',  # 2+2, 5*3, etc.
            r'(sin|cos|tan|log|sqrt|exp|abs|factorial)\s*\(',  # Funciones matemáticas
            r'\d+\s*\*\*\s*\d+',  # Potencias con **
            r'\d+\s*\^\s*\d+',    # Potencias con ^
            r'sqrt\(\d+\)',       # Raíces cuadradas
            r'\bpi\b|\be\b',      # Constantes matemáticas
            r'\d+\.\d+',          # Números decimales en operaciones
            r'\(\s*\d+.*?\)',     # Expresiones con paréntesis
        ]
        
        # También verificar frases que indican cálculo
        calc_phrases = [
            'cuanto es', 'cuánto es', 'calcula', 'resuelve', 'resultado de'
        ]
        
        has_math_pattern = any(re.search(pattern, cleaned) for pattern in math_patterns)
        has_calc_phrase = any(phrase in cleaned for phrase in calc_phrases)
        
        return has_math_pattern or has_calc_phrase
    
    def safe_eval(self, expression):
        """Evaluación segura de expresiones matemáticas"""
        try:
            # Limpiar la expresión
            expression = str(expression).strip()
            
            # Reemplazos para compatibilidad
            replacements = {
                '^': '**',      # Potencias
                '×': '*',       # Multiplicación
                '÷': '/',       # División
                '√': 'sqrt',    # Raíz cuadrada
                'π': str(math.pi),  # Pi
                ' x ': '*',     # Multiplicación con espacios
                'sen': 'sin',   # Seno en español
                'ln': 'log',    # Logaritmo natural
            }
            
            for old, new in replacements.items():
                expression = expression.replace(old, new)
            
            # Agregar math. a las funciones si no lo tienen
            math_funcs = ['sin', 'cos', 'tan', 'log', 'sqrt', 'exp', 'abs', 'factorial', 'ceil', 'floor']
            for func_name in math_funcs:
                pattern = rf'\b{func_name}\s*\('
                if re.search(pattern, expression) and f'math.{func_name}' not in expression:
                    expression = re.sub(pattern, f'math.{func_name}(', expression)
            
            # Reemplazar constantes
            expression = expression.replace('pi', str(math.pi))
            expression = expression.replace('e', str(math.e))
            
            # Evaluación segura
            allowed_names = {
                "__builtins__": {},
                "math": math,
                **self.variables
            }
            
            result = eval(expression, allowed_names)
            return result
            
        except Exception as e:
            raise ValueError(f"Error evaluando '{expression}': {str(e)}")
    
    def format_number(self, num):
        """Formatear números de forma elegante"""
        if isinstance(num, (int, float)):
            if num == int(num):
                return str(int(num))
            elif abs(num) > 1000000:
                # Notación científica para números muy grandes
                return f"{num:.3e}"
            else:
                # Mostrar hasta 10 decimales, eliminando ceros
                formatted = f"{num:.10f}".rstrip('0').rstrip('.')
                return formatted
        return str(num)
    
    def extract_calculation(self, message):
        """Extraer expresión matemática de mensajes en lenguaje natural"""
        # Patrones para extraer la expresión matemática
        patterns = [
            r'(?:cuanto es|cuánto es|calcula|resuelve|dame el resultado de|resultado de)\s*(.+)',
            r'^(.+)$'  # Si no encuentra patrón específico, toma todo
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message.strip(), re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                # Limpiar palabras innecesarias al final
                words_to_remove = ['por favor', 'porfavor', 'gracias', '?', '¿', '!', '¡']
                for word in words_to_remove:
                    extracted = extracted.replace(word, '').strip()
                return extracted
        
        return message.strip()
    
    def get_ai_response_sync(self, message):
        """Obtener respuesta de la IA de forma síncrona con rate limiting mejorado"""
        if not self.ai_available:
            return None
        
        # === RATE LIMITING INTELIGENTE ===
        current_time = time.time()
        time_since_last = current_time - self.last_api_call
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            print(f"⏳ Rate limiting: esperando {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        # === RETRY LOGIC CON BACKOFF EXPONENCIAL ===
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.last_api_call = time.time()
                
                # Si es un retry, agregar delay adicional
                if attempt > 0:
                    retry_delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"🔄 Reintento {attempt + 1}/{max_retries} después de {retry_delay:.1f}s")
                    time.sleep(retry_delay)
                    self.last_api_call = time.time()  # Actualizar después del delay
                
                # Crear prompt especializado en matemáticas
                system_prompt = """Eres un profesor de matemáticas experto, amigable y conversacional. Tu trabajo es:

1. Responder preguntas matemáticas con explicaciones claras y educativas
2. Explicar conceptos matemáticos de forma comprensible y profunda
3. Ayudar con problemas paso a paso cuando sea necesario
4. Ser natural y conversacional, como un profesor universitario que realmente se preocupa por que el estudiante entienda
5. Dar contexto histórico, aplicaciones prácticas o curiosidades cuando sea relevante
6. No usar demasiados emojis, mantener un tono profesional pero amigable

Si el usuario hace una pregunta matemática, explica tanto el "qué" como el "por qué". 
Si pregunta sobre conceptos, da explicaciones profundas pero accesibles.
Si es una conversación general relacionada con matemáticas, mantén el contexto educativo."""

                # Construir contexto de conversación si existe
                conversation_context = ""
                if self.conversation_history:
                    recent_messages = self.conversation_history[-3:]  # Últimos 3 intercambios
                    conversation_context = "\n\nContexto de conversación previa:\n"
                    for user_msg, bot_response in recent_messages:
                        conversation_context += f"Usuario: {user_msg}\nAsistente: {bot_response[:150]}...\n"
                
                full_prompt = f"{system_prompt}\n\nPregunta actual del usuario: {message}{conversation_context}"
                
                # Payload optimizado para rate limiting
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": full_prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topK": 20,  # Reducido para menos carga
                        "topP": 0.9,  # Reducido para menos carga
                        "maxOutputTokens": 512,  # Reducido para respuestas más rápidas
                        "candidateCount": 1
                    },
                    "safetySettings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH", 
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        }
                    ]
                }
                
                response = requests.post(
                    self.gemini_url,
                    headers={
                        'Content-Type': 'application/json',
                        'x-goog-api-key': self.api_key,
                        'User-Agent': 'MathChatBot/1.0'
                    },
                    json=payload,
                    timeout=20  # Timeout aumentado para dar más tiempo
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'candidates' in data and len(data['candidates']) > 0:
                        candidate = data['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            ai_response = candidate['content']['parts'][0]['text']
                            return ai_response.strip()
                    
                    print("⚠️ Respuesta inesperada de la IA")
                    return None
                    
                elif response.status_code == 429:
                    print(f"⚠️ Rate limit en intento {attempt + 1}/{max_retries}")
                    if attempt == max_retries - 1:
                        print("❌ Rate limit persistente - la IA funcionará intermitentemente")
                        return None
                    # Continuar al siguiente intento con backoff
                    continue
                    
                else:
                    print(f"❌ Error en Gemini API: {response.status_code}")
                    try:
                        error_detail = response.json()
                        print(f"📄 Detalle del error: {error_detail}")
                    except:
                        print(f"📄 Response text: {response.text[:200]}")
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"⚠️ Timeout en intento {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                    print("❌ Timeout persistente con la IA")
                    return None
                continue
                
            except Exception as e:
                print(f"❌ Error en intento {attempt + 1}/{max_retries}: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                continue
        
        return None
    
    def get_fallback_response(self, message):
        """Respuestas de emergencia cuando la IA no está disponible"""
        message_lower = message.lower().strip()
        
        # Respuestas para saludos
        if any(word in message_lower for word in ['hola', 'buenos días', 'buenas tardes', 'hey', 'saludos']):
            responses = [
                "¡Hola! Soy tu asistente matemático con IA y gráficas. ¿En qué puedo ayudarte?",
                "¡Saludos! Puedo resolver problemas matemáticos y crear gráficas. ¿Qué necesitas?",
                "¡Bienvenido! Estoy listo para matemáticas y visualizaciones. ¿Empezamos?"
            ]
            return random.choice(responses)
        
        # Respuestas para ayuda
        if any(word in message_lower for word in ['ayuda', 'help', 'qué puedes hacer']):
            return """Soy tu asistente matemático inteligente con capacidades de gráficas. Puedo:

**🧮 Cálculos:** Operaciones, funciones trigonométricas, logaritmos
**📊 Gráficas:** "grafica sin(x)", "grafica 3x+2", "f(x) = x^2+1"
**📚 Conceptos:** Explicaciones profundas de temas matemáticos
**💬 Conversación:** Charlas educativas sobre matemáticas

**Ejemplos de gráficas:**
• "grafica sin(x)"
• "grafica 3x+2"
• "f(x) = x^2+3x+1"
• "compara sin(x) vs cos(x)"
• "dibuja 2x-5"

¿Qué te gustaría explorar?"""
        
        return f"Entiendo que preguntas sobre '{message}'. Puedo ayudarte con cálculos, conceptos matemáticos y crear gráficas de funciones. ¿Podrías ser más específico?"
    
    def get_response(self, message):
        """Método principal para obtener respuestas del chatbot"""
        try:
            if not message or not message.strip():
                return {
                    'response': "¡Hola! Soy tu asistente matemático con IA y gráficas. ¿En qué puedo ayudarte?",
                    'type': 'text'
                }
            
            # Mantener contexto de conversación
            self.context.append(message.strip())
            if len(self.context) > 10:
                self.context.pop(0)
            
            print(f"🤔 Analizando: {message}")
            
            # PASO 1: Verificar si es una solicitud de gráfica
            if self.is_chart_request(message):
                print("📊 Detectada solicitud de gráfica")
                
                chart_info = self.parse_chart_request(message)
                print(f"📋 Info de gráfica: {chart_info}")
                
                if chart_info['functions']:
                    chart_data = self.generate_chart_data(chart_info)
                    
                    if chart_data:
                        # Usar IA para explicar la gráfica si está disponible
                        function_names = []
                        for f in chart_info['functions']:
                            if isinstance(f, dict):
                                if f['type'] == 'algebraic':
                                    function_names.append(f['original'])
                                else:
                                    function_names.append(f['name'])
                            else:
                                function_names.append(f)
                        
                        if self.ai_available:
                            ai_explanation = self.get_ai_response_sync(
                                f"El usuario pidió graficar {', '.join(function_names)}. Explica brevemente estas funciones matemáticas y sus características principales."
                            )
                            response_text = ai_explanation if ai_explanation else f"Aquí tienes la gráfica de {', '.join(function_names)}."
                        else:
                            response_text = f"Aquí tienes la gráfica de {', '.join(function_names)}. (IA no disponible para explicación detallada)"
                        
                        # Guardar en historial
                        self.conversation_history.append((message, response_text))
                        if len(self.conversation_history) > 5:
                            self.conversation_history.pop(0)
                        
                        return {
                            'response': response_text,
                            'type': 'chart',
                            'chart_data': chart_data
                        }
                    else:
                        return {
                            'response': "Lo siento, hubo un problema generando la gráfica. ¿Podrías intentar con una función más simple?",
                            'type': 'text'
                        }
                else:
                    return {
                        'response': "No pude identificar qué función graficar. Intenta con: 'grafica sin(x)', 'grafica 3x+2', 'f(x) = x^2+1', etc.",
                        'type': 'text'
                    }
            
            # PASO 2: Verificar si es una expresión matemática calculable
            elif self.is_mathematical_expression(message):
                print("🧮 Detectada expresión matemática")
                try:
                    # Extraer la expresión matemática
                    expression = self.extract_calculation(message)
                    print(f"📝 Expresión extraída: {expression}")
                    
                    # Calcular resultado
                    result = self.safe_eval(expression)
                    self.last_result = result
                    formatted_result = self.format_number(result)
                    
                    print(f"✅ Resultado calculado: {formatted_result}")
                    
                    # Usar IA para enriquecer la respuesta con explicación si está disponible
                    if self.ai_available:
                        ai_explanation = self.get_ai_response_sync(
                            f"El usuario calculó '{expression}' y obtuve como resultado {formatted_result}. Explica brevemente esta operación matemática y proporciona contexto educativo relevante. Sé conciso pero informativo."
                        )
                        
                        if ai_explanation:
                            response_text = f"**Resultado:** {formatted_result}\n\n{ai_explanation}"
                        else:
                            # Explicación básica si la IA falla
                            response_text = f"**Resultado:** {formatted_result}\n\nEste es el resultado de calcular: {expression}"
                    else:
                        # Explicación básica si IA no está disponible
                        response_text = f"**Resultado:** {formatted_result}\n\nEste es el resultado de calcular: {expression} (IA no disponible para explicación detallada)"
                    
                    # Guardar en historial
                    self.conversation_history.append((message, response_text))
                    if len(self.conversation_history) > 5:
                        self.conversation_history.pop(0)
                    
                    return {
                        'response': response_text,
                        'type': 'calculation'
                    }
                    
                except Exception as e:
                    print(f"⚠️ Error en cálculo: {str(e)}")
                    # Si falla el cálculo, dejar que la IA maneje todo
                    pass
            
            # PASO 3: Usar IA para respuesta conversacional
            if self.ai_available:
                print("🧠 Consultando IA...")
                ai_response = self.get_ai_response_sync(message)
                
                if ai_response:
                    print("✅ Respuesta de IA obtenida")
                    # Guardar en historial
                    self.conversation_history.append((message, ai_response))
                    if len(self.conversation_history) > 5:
                        self.conversation_history.pop(0)
                    return {
                        'response': ai_response,
                        'type': 'conversation'
                    }
            
            # PASO 4: Usar respuestas de fallback
            print("⚠️ IA no disponible, usando fallback")
            fallback_response = self.get_fallback_response(message)
            return {
                'response': fallback_response,
                'type': 'fallback'
            }
                
        except Exception as e:
            print(f"❌ Error general en get_response: {str(e)}")
            return {
                'response': "Disculpa, hubo un problema técnico. ¿Puedes intentar reformular tu pregunta?",
                'type': 'error'
            }
    
    def reset_context(self):
        """Limpiar contexto y historial de conversación"""
        self.context = []
        self.conversation_history = []
        self.variables = {}
        self.last_result = None
        return "Perfecto, he iniciado una nueva conversación. ¿En qué puedo ayudarte ahora?"
    
    def get_conversation_stats(self):
        """Obtener estadísticas de la conversación actual"""
        return {
            'messages_in_context': len(self.context),
            'conversation_history': len(self.conversation_history),
            'last_result': self.last_result,
            'variables_defined': len(self.variables),
            'ai_available': self.ai_available
        }