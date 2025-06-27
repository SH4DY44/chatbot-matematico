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
        
        # Rate limiting más relajado para Groq - 2 segundos
        self.last_api_call = 0
        self.min_interval = 2.0  # Groq es mucho más permisivo
        
        # API Key - Ahora soporta múltiples proveedores
        self.api_key = os.getenv('GROQ_API_KEY') or os.getenv('GEMINI_API_KEY')
        self.api_provider = 'groq' if os.getenv('GROQ_API_KEY') else 'gemini'
        
        if not self.api_key:
            print("⚠️ ADVERTENCIA: No se encontró GROQ_API_KEY ni GEMINI_API_KEY.")
        else:
            if self.api_provider == 'groq':
                print("🚀 Usando Groq API (más rápida y permisiva)")
                self.api_url = "https://api.groq.com/openai/v1/chat/completions"
            else:
                print("🧠 Usando Gemini API (con limitaciones)")
                self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        
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
        print("✅ ChatBot completo listo!")
    
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
    
    def natural_language_to_expression(self, message):
        """Convierte frases comunes en español a notación algebraica"""
        msg = message.lower()
        # Potencias
        msg = re.sub(r'x\s*al\s*cuadrado', 'x^2', msg)
        msg = re.sub(r'x\s*al\s*cubo', 'x^3', msg)
        msg = re.sub(r'x\s*a\s*la\s*quarta', 'x^4', msg)
        msg = re.sub(r'x\s*a\s*la\s*quinta', 'x^5', msg)
        msg = re.sub(r'x\s*elevado\s*a\s*(\d+)', r'x^\1', msg)
        # Sumas y restas
        msg = re.sub(r'más', '+', msg)
        msg = re.sub(r'menos', '-', msg)
        # Multiplicaciones implícitas
        msg = re.sub(r'(\d+)\s*x', r'\1x', msg)
        # Eliminar palabras innecesarias
        msg = re.sub(r'grafica?r?\s*', '', msg)
        msg = re.sub(r'dibuja?r?\s*', '', msg)
        msg = re.sub(r'la\s*función\s*', '', msg)
        msg = re.sub(r'función\s*', '', msg)
        msg = re.sub(r'\s+', ' ', msg)
        return msg.strip()
    
    def parse_chart_request(self, message):
        """Analizar qué tipo de gráfica quiere el usuario, soportando frases naturales y múltiples funciones"""
        message = self.natural_language_to_expression(message)
        message_lower = message.lower()
        
        print(f"📊 Parseando solicitud de gráfica: {message}")
        
        # Separar múltiples funciones por coma, 'y', 'vs', 'versus'
        split_patterns = [',', ' y ', ' vs ', ' versus ', ' contra ']
        for pat in split_patterns:
            if pat in message:
                parts = [p.strip() for p in message.split(pat) if p.strip()]
                break
        else:
            parts = [message]
        
        functions = []
        for part in parts:
            print(f"🔍 Analizando parte: {part}")
            
            # Buscar patrones algebraicos primero
            algebraic_patterns = [
                r'f\(x\)\s*=\s*([^,\.\!?]+)',  # f(x) = 3x+2
                r'([\-\+]?\d*\.?\d*x(\*\*\d+|\^\d+)?([\+\-]\d+)?)',  # 3x+2, -x^2+1, etc.
                r'(sin|cos|tan|log|exp|sqrt|abs)\s*\([^)]*\)',  # sin(x), log(x+1), etc.
                r'x\*\*\d+',  # x**2, x**3, etc.
                r'x\^\d+',   # x^2, x^3, etc.
                r'\bx\b',    # x solo
            ]
            
            found = False
            for pattern in algebraic_patterns:
                match = re.search(pattern, part)
                if match:
                    expr = match.group(0).strip()
                    if 'f(x)' in expr:
                        expr = expr.split('=')[1].strip()
                    normalized = self.normalize_function(expr)
                    print(f"✅ Función algebraica encontrada: {expr} -> {normalized}")
                    functions.append({
                        'type': 'algebraic',
                        'expression': normalized,
                        'original': expr
                    })
                    found = True
                    break
            
            # Si no es algebraica, probar predefinidas
            if not found:
                predefs = {
                    'sin(x)': 'sin', 'cos(x)': 'cos', 'tan(x)': 'tan', 
                    'log(x)': 'log', 'exp(x)': 'exp', 'sqrt(x)': 'sqrt', 
                    'abs(x)': 'abs', 'x^2': 'x^2', 'x^3': 'x^3',
                    'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
                    'log': 'log', 'exp': 'exp'
                }
                for key, val in predefs.items():
                    if key in part:
                        print(f"✅ Función predefinida encontrada: {val}")
                        functions.append({'type': 'predefined', 'name': val})
                        found = True
                        break
        
        # Rango
        range_match = re.search(r'de\s+(-?\d+)\s+a\s+(-?\d+)', message_lower)
        if range_match:
            x_range = [int(range_match.group(1)), int(range_match.group(2))]
        else:
            x_range = [-10, 10]
        
        # Comparación
        is_comparison = len(functions) > 1
        
        result = {
            'functions': functions,
            'range': x_range,
            'is_comparison': is_comparison,
            'type': 'comparison' if is_comparison else 'single'
        }
        
        print(f"📋 Resultado del parsing: {result}")
        return result
    
    def is_algebraic_function(self, function_str):
        """Verificar si una cadena es una función algebraica válida"""
        # Debe contener 'x' para ser una función
        if 'x' not in function_str:
            return False
        
        # Verificar caracteres permitidos
        allowed_chars = set('0123456789x+-*/^(). ')
        if not all(c in allowed_chars for c in function_str):
            return False
        
        return True
    
    def generate_chart_data(self, chart_info):
        """Generar datos para la gráfica"""
        try:
            print(f"📊 Generando datos de gráfica para: {chart_info}")
            
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
                print(f"🔧 Procesando función {i+1}: {func_info}")
                y_values = []
                label = "función"
                
                # Determinar el tipo de función
                if isinstance(func_info, dict):
                    if func_info['type'] == 'algebraic':
                        # Función algebraica personalizada
                        func_expression = func_info['expression']
                        label = func_info['original']
                        
                        print(f"📝 Evaluando expresión algebraica: {func_expression}")
                        
                        for x in x_values:
                            try:
                                # Reemplazar x en la expresión
                                expression = func_expression.replace('x', f'({x})')
                                # Añadir math. a las funciones
                                expression = re.sub(r'\b(sin|cos|tan|log|exp|sqrt|abs)\b', r'math.\1', expression)
                                
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
                        label = func_name + '(x)' if func_name not in ['x^2', 'x^3'] else func_name
                        
                        print(f"📝 Evaluando función predefinida: {func_name}")
                        
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
                print(f"✅ Dataset creado para {label} con {len(dataset['data'])} puntos")
            
            chart_data = {
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
            
            print(f"✅ Datos de gráfica generados exitosamente")
            return chart_data
            
        except Exception as e:
            print(f"❌ Error generando datos de gráfica: {str(e)}")
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
        """Obtener respuesta de IA usando Groq (más permisivo) o Gemini como fallback"""
        if not self.api_key:
            return None
        
        # Rate limiting más relajado - 2 segundos para Groq
        current_time = time.time()
        time_since_last = current_time - self.last_api_call
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            print(f"⏳ Esperando {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        try:
            self.last_api_call = time.time()
            
            if self.api_provider == 'groq':
                return self.use_groq_api(message)
            else:
                return self.use_gemini_api(message)
                
        except Exception as e:
            print(f"❌ Error con IA: {str(e)}")
            return None
    
    def use_groq_api(self, message):
        """Usar Groq API (mucho más permisiva que Gemini)"""
        try:
            payload = {
                "model": "llama-3.1-8b-instant",  # Modelo gratuito muy rápido
                "messages": [
                    {
                        "role": "system", 
                        "content": "Eres un profesor de matemáticas experto y amigable. Explica conceptos claramente y proporciona ejemplos útiles. Sé conciso pero educativo."
                    },
                    {
                        "role": "user", 
                        "content": message
                    }
                ],
                "max_tokens": 400,
                "temperature": 0.7,
                "stream": False
            }
            
            response = requests.post(
                self.api_url,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    ai_response = data['choices'][0]['message']['content']
                    print("✅ IA respondió exitosamente (Groq)")
                    return ai_response.strip()
            
            elif response.status_code == 429:
                print("⚠️ Rate limit en Groq")
                return None
            else:
                print(f"❌ Error Groq {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"⚠️ Error con Groq: {str(e)}")
            return None
    
    def use_gemini_api(self, message):
        """Fallback: Usar Gemini API (más restrictiva)"""
        try:
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"Como profesor de matemáticas, responde: {message[:150]}"
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 300
                }
            }
            
            response = requests.post(
                self.api_url,
                headers={
                    'Content-Type': 'application/json'
                },
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'candidates' in data and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        ai_response = candidate['content']['parts'][0]['text']
                        print("✅ IA respondió exitosamente (Gemini)")
                        return ai_response.strip()
            
            elif response.status_code == 429:
                print("⚠️ Rate limit en Gemini")
                return None
            else:
                print(f"❌ Error Gemini {response.status_code}")
                return None
                
        except Exception as e:
            print(f"⚠️ Error con Gemini: {str(e)}")
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
    
    def is_conceptual_request(self, message):
        """Detectar si el usuario pide una explicación, ejemplo o concepto matemático"""
        conceptual_keywords = [
            'ejemplo', 'ejemplos', 'explica', 'explicación', 'qué es', 'que es', 'definición',
            'derivada', 'integral', 'teorema', 'propiedad', 'demuestra', 'demostración',
            'cómo funciona', 'como funciona', 'para qué sirve', 'para que sirve', 'aplicación', 'aplicaciones',
            'historia', 'origen', 'uso', 'usos', 'ventaja', 'ventajas', 'desventaja', 'desventajas',
            'característica', 'características', 'concepto', 'introducción', 'fundamento', 'fundamentos',
            'explicame', 'describe', 'descripción', 'ejercicios resueltos', 'ejercicio resuelto', 'ejercicio', 'ejercicios'
        ]
        message_lower = message.lower()
        return any(word in message_lower for word in conceptual_keywords)
    
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

            # PASO 0: Verificar si es una solicitud conceptual o de ejemplos
            if self.is_conceptual_request(message):
                print("📚 Solicitud conceptual detectada")
                if self.api_key:
                    ai_explanation = self.get_ai_response_sync(
                        f"Da una explicación clara, breve y con ejemplos sobre: {message}"
                    )
                    response_text = ai_explanation if ai_explanation else "Aquí tienes una explicación sobre el tema solicitado."
                else:
                    response_text = "Puedo darte ejemplos y explicaciones de temas matemáticos. ¿Sobre qué tema específico quieres saber más?"
                self.conversation_history.append((message, response_text))
                if len(self.conversation_history) > 5:
                    self.conversation_history.pop(0)
                return {
                    'response': response_text,
                    'type': 'concept'
                }

            # PASO 1: Verificar si es una solicitud de gráfica
            if self.is_chart_request(message):
                print("📊 Detectada solicitud de gráfica")
                
                chart_info = self.parse_chart_request(message)
                print(f"📋 Info de gráfica: {chart_info}")
                
                if chart_info['functions']:
                    chart_data = self.generate_chart_data(chart_info)
                    
                    if chart_data:
                        function_names = []
                        for f in chart_info['functions']:
                            if isinstance(f, dict):
                                if f['type'] == 'algebraic':
                                    function_names.append(f['original'])
                                else:
                                    function_names.append(f['name'])
                            else:
                                function_names.append(f)
                        
                        # Intentar explicación con IA
                        if self.api_key:
                            ai_explanation = self.get_ai_response_sync(
                                f"Explica brevemente la función matemática {', '.join(function_names)} y sus características principales."
                            )
                            response_text = ai_explanation if ai_explanation else f"Aquí tienes la gráfica de {', '.join(function_names)}."
                        else:
                            response_text = f"Aquí tienes la gráfica de {', '.join(function_names)}."
                        
                        # Guardar en historial
                        self.conversation_history.append((message, response_text))
                        if len(self.conversation_history) > 5:
                            self.conversation_history.pop(0)
                        
                        print("✅ Enviando datos de gráfica al frontend")
                        return {
                            'response': response_text,
                            'type': 'chart',
                            'chart_data': chart_data
                        }
                    else:
                        return {
                            'response': "Hubo un problema generando la gráfica. ¿Podrías intentar con una función más simple como 'grafica sin(x)' o 'grafica x^2'?",
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
                    
                    # Intentar explicación con IA
                    if self.api_key:
                        ai_explanation = self.get_ai_response_sync(
                            f"El usuario calculó '{expression}' = {formatted_result}. Explica brevemente esta operación matemática."
                        )
                        
                        if ai_explanation:
                            response_text = f"**Resultado:** {formatted_result}\n\n{ai_explanation}"
                        else:
                            response_text = f"**Resultado:** {formatted_result}"
                    else:
                        response_text = f"**Resultado:** {formatted_result}"
                    
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
                    # Si falla el cálculo, continuar a IA
                    pass
            
            # PASO 3: IA Conversacional
            if self.api_key:
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
                else:
                    print("⚠️ IA no respondió, usando fallback")
            
            # PASO 4: Fallback cuando no hay IA disponible
            print("⚠️ Usando fallback")
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
            'ai_available': bool(self.api_key)
        }