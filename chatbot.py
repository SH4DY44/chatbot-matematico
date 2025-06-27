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
        
        # Rate limiting ULTRA conservador - 15 segundos entre llamadas
        self.last_api_call = 0
        self.min_interval = 15.0  # 15 segundos para respetar límites de Google
        
        # API Key de Google Gemini - SOLO desde variable de entorno
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            print("⚠️ ADVERTENCIA: GEMINI_API_KEY no configurada. IA funcionará en modo fallback.")
        else:
            self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
            print("🧠 IA configurada y lista para usar")
        
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
    
    def parse_chart_request(self, message):
        """Analizar qué tipo de gráfica quiere el usuario"""
        message_lower = message.lower()
        
        # Detectar funciones específicas
        functions = []
        
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
        
        # Funciones predefinidas si no se encontró algebraica
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
        """Obtener respuesta de la IA con rate limiting ULTRA conservador"""
        if not self.api_key:
            return None
        
        # Rate limiting de 15 segundos mínimo
        current_time = time.time()
        time_since_last = current_time - self.last_api_call
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            print(f"⏳ Esperando {sleep_time:.0f}s para respetar rate limit de Google")
            time.sleep(sleep_time)
        
        try:
            self.last_api_call = time.time()
            
            # Payload MÍNIMO para reducir carga
            payload = {
                "contents": [{
                    "parts": [{
                        "text": message[:200]  # Limitar a 200 caracteres
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.5,
                    "maxOutputTokens": 150  # Respuestas muy cortas
                }
            }
            
            response = requests.post(
                self.gemini_url,
                headers={
                    'Content-Type': 'application/json',
                    'x-goog-api-key': self.api_key
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'candidates' in data and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        ai_response = candidate['content']['parts'][0]['text']
                        print("✅ IA respondió exitosamente")
                        return ai_response.strip()
            
            elif response.status_code == 429:
                print("⚠️ Rate limit - Google necesita más tiempo entre requests")
                return None
            else:
                print(f"❌ Error {response.status_code}: {response.text[:100]}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
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
                        function_names = []
                        for f in chart_info['functions']:
                            if isinstance(f, dict):
                                if f['type'] == 'algebraic':
                                    function_names.append(f['original'])
                                else:
                                    function_names.append(f['name'])
                            else:
                                function_names.append(f)
                        
                        response_text = f"Aquí tienes la gráfica de {', '.join(function_names)}."
                        
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
            
            # PASO 3: IA Conversacional CON INFORMACIÓN DE TIMING
            if self.api_key:
                print("🧠 Consultando IA...")
                
                # Verificar si hace poco se usó la IA
                time_since_last = time.time() - self.last_api_call
                if time_since_last < self.min_interval:
                    remaining = self.min_interval - time_since_last
                    return {
                        'response': f"🤖 IA disponible en {remaining:.0f} segundos debido a límites de Google. Mientras tanto, puedo ayudarte con cálculos matemáticos (ej: 2+2) o gráficas (ej: grafica sin(x)).",
                        'type': 'rate_limit_info'
                    }
                
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
                    return {
                        'response': "🤖 IA temporalmente limitada por Google. Puedo ayudarte con cálculos (2+2) o gráficas (grafica sin(x)).",
                        'type': 'ai_limited'
                    }
            
            # PASO 4: Fallback cuando no hay API key
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