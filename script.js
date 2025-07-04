class MathChatBot {
    constructor() {
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.status = document.getElementById('status');
        
        // Paneles de la interfaz
        this.welcomePanel = document.getElementById('welcomePanel');
        this.categoriesPanel = document.getElementById('categoriesPanel');
        this.examplesPanel = document.getElementById('examplesPanel');
        this.chartsPanel = document.getElementById('chartsPanel');
        this.mathKeyboard = document.getElementById('mathKeyboard');
        this.suggestions = document.getElementById('suggestions');
        
        // Botones de herramientas
        this.keyboardToggle = document.getElementById('keyboardToggle');
        this.examplesToggle = document.getElementById('examplesToggle');
        this.chartsToggle = document.getElementById('chartsToggle');
        this.helpToggle = document.getElementById('helpToggle');
        
        // Estado de la interfaz
        this.keyboardVisible = false;
        this.examplesVisible = false;
        this.chartsVisible = false;
        this.currentCategory = 'basico';
        this.welcomePanelHidden = false; 
        
        // Contador para gráficas únicas
        this.chartCount = 0;
        
        // URLs del backend 
        this.apiUrl = this.getApiUrl() + '/api/chat';
        this.healthUrl = this.getApiUrl() + '/health';
        
        // Ejemplos por categoría (incluyendo gráficas)
        this.examples = {
            basico: {
                name: 'Aritmética',
                items: ['2+2', '15*7', '100/4', '12-8', '25*4+10', '144/12*3', '2**8', 'sqrt(16)']
            },
            funciones: {
                name: 'Funciones',
                items: ['sin(pi/2)', 'cos(0)', 'log(100)', 'exp(1)', 'sqrt(25)', 'factorial(5)', 'abs(-10)', 'ceil(4.2)']
            },
            conceptos: {
                name: 'Conceptos',
                items: ['qué es pi', 'qué es el número e', 'explicar logaritmos', 'para qué sirve seno', 'qué es derivada', 'qué es integral']
            },
            graficas: {
                name: 'Gráficas',
                items: ['grafica sin(x)', 'grafica x^2', 'grafica 3x+2', 'grafica log(x)', 'compara sin vs cos', 'f(x) = 2x-1']
            }
        };
        
        // Sugerencias automáticas
        this.suggestions_data = [
            '2+2', 'sqrt(16)', 'sin(pi/2)', 'qué es pi',
            'log(100)', 'factorial(5)', 'grafica sin(x)', 'grafica x^2',
            'cuánto es 15*7', 'cos(0)', 'exp(1)', 'grafica 3x+2'
        ];
        
        this.init();
    }
    
    getApiUrl() {
        // No se te vaya a olvidar hacer la comunicacion con la API
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return 'http://localhost:5000';
        } else {
            return window.location.origin;
        }
    }
    
    init() {
        this.setWelcomeTime();
        this.setupEventListeners();
        this.checkConnection();
        
        // Verificar conexión cada 30 segundos
        setInterval(() => this.checkConnection(), 30000);
        
        // Auto-focus al input
        this.messageInput.focus();
    }
    
    setupEventListeners() {
        // Eventos del chat
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // NUEVO: Ocultar panel de bienvenida cuando usuario empiece a escribir
        this.messageInput.addEventListener('input', (e) => {
            this.showSuggestions(e.target.value);
            
            // Si hay texto y el panel de bienvenida aún está visible, ocultarlo
            if (e.target.value.trim().length > 0 && !this.welcomePanelHidden) {
                this.hideWelcomePanelSmooth();
            }
        });
        
        // Ocultar sugerencias al hacer click fuera
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.chat-input')) {
                this.hideSuggestions();
            }
        });
    }
    
    setWelcomeTime() {
        const welcomeTimeEl = document.getElementById('welcomeTime');
        if (welcomeTimeEl) {
            welcomeTimeEl.textContent = new Date().toLocaleTimeString('es-ES', {
                hour: '2-digit',
                minute: '2-digit'
            });
        }
    }
    
    // === MANEJO DE PANELES ===
    
    // NUEVO: Ocultar panel de bienvenida con transición suave
    hideWelcomePanelSmooth() {
        if (this.welcomePanelHidden) return;
        
        this.welcomePanelHidden = true;
        this.welcomePanel.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
        this.welcomePanel.style.opacity = '0';
        this.welcomePanel.style.transform = 'translateY(-20px)';
        
        setTimeout(() => {
            this.welcomePanel.style.display = 'none';
        }, 500);
    }
    
    selectCategory(type) {
        this.hideWelcomePanel();
        
        switch(type) {
            case 'ejemplos':
                this.showCategories();
                break;
            case 'graficas':
                this.showCharts();
                break;
            case 'escribir':
                this.messageInput.focus();
                break;
        }
    }
    
    hideWelcomePanel() {
        this.welcomePanel.style.display = 'none';
        this.welcomePanelHidden = true;
    }
    
    showCategories() {
        this.hideAllPanels();
        this.categoriesPanel.style.display = 'block';
        this.examplesToggle.classList.add('active');
        this.examplesVisible = true;
    }
    
    showCharts() {
        this.hideAllPanels();
        this.chartsPanel.style.display = 'block';
        this.chartsToggle.classList.add('active');
        this.chartsVisible = true;
    }
    
    showExamples(category) {
        this.currentCategory = category;
        
        if (category === 'graficas') {
            this.showCharts();
            return;
        }
        
        this.hideAllPanels();
        this.examplesPanel.style.display = 'block';
        
        // Actualizar título y ejemplos
        document.getElementById('currentCategory').textContent = this.examples[category].name;
        this.renderExamples(category);
        
        this.examplesToggle.classList.add('active');
        this.examplesVisible = true;
    }
    
    renderExamples(category) {
        const grid = document.getElementById('examplesGrid');
        const examples = this.examples[category].items;
        
        grid.innerHTML = examples.map(example => 
            `<button class="example-btn" onclick="sendExample('${example}')">${example}</button>`
        ).join('');
    }
    
    hideAllPanels() {
        this.categoriesPanel.style.display = 'none';
        this.examplesPanel.style.display = 'none';
        this.chartsPanel.style.display = 'none';
        this.mathKeyboard.style.display = 'none';
        
        // Resetear botones
        this.keyboardToggle.classList.remove('active');
        this.examplesToggle.classList.remove('active');
        this.chartsToggle.classList.remove('active');
        
        this.keyboardVisible = false;
        this.examplesVisible = false;
        this.chartsVisible = false;
    }
    
    // === TECLADO MATEMÁTICO ===
    
    toggleKeyboard() {
        if (this.keyboardVisible) {
            this.mathKeyboard.style.display = 'none';
            this.keyboardToggle.classList.remove('active');
            this.keyboardVisible = false;
        } else {
            this.hideAllPanels();
            this.mathKeyboard.style.display = 'block';
            this.keyboardToggle.classList.add('active');
            this.keyboardVisible = true;
        }
    }
    
    toggleExamples() {
        if (this.examplesVisible) {
            this.hideAllPanels();
        } else {
            this.showCategories();
        }
    }
    
    toggleCharts() {
        if (this.chartsVisible) {
            this.hideAllPanels();
        } else {
            this.showCharts();
        }
    }
    
    insertSymbol(symbol) {
        const cursorPos = this.messageInput.selectionStart;
        const currentValue = this.messageInput.value;
        
        const newValue = currentValue.slice(0, cursorPos) + symbol + currentValue.slice(cursorPos);
        this.messageInput.value = newValue;
        
        // Mover cursor después del símbolo insertado
        this.messageInput.setSelectionRange(cursorPos + symbol.length, cursorPos + symbol.length);
        this.messageInput.focus();
        
        // Trigger input event para activar hideWelcomePanel si es necesario
        this.messageInput.dispatchEvent(new Event('input'));
    }
    
    insertFunction(func) {
        const cursorPos = this.messageInput.selectionStart;
        const currentValue = this.messageInput.value;
        
        const newValue = currentValue.slice(0, cursorPos) + func + currentValue.slice(cursorPos);
        this.messageInput.value = newValue;
        
        // Mover cursor dentro de los paréntesis
        const newCursorPos = cursorPos + func.length - 1;
        this.messageInput.setSelectionRange(newCursorPos, newCursorPos);
        this.messageInput.focus();
        
        // Trigger input event
        this.messageInput.dispatchEvent(new Event('input'));
    }
    
    clearInput() {
        this.messageInput.value = '';
        this.messageInput.focus();
        this.hideSuggestions();
    }
    
    // === SUGERENCIAS ===
    
    showSuggestions(inputValue) {
        if (!inputValue || inputValue.length < 2) {
            this.hideSuggestions();
            return;
        }
        
        const filtered = this.suggestions_data.filter(suggestion =>
            suggestion.toLowerCase().includes(inputValue.toLowerCase())
        );
        
        if (filtered.length > 0) {
            this.renderSuggestions(filtered.slice(0, 4));
        } else {
            this.hideSuggestions();
        }
    }
    
    renderSuggestions(suggestions) {
        this.suggestions.innerHTML = suggestions.map(suggestion =>
            `<div class="suggestion-item" onclick="selectSuggestion('${suggestion}')">${suggestion}</div>`
        ).join('');
        
        this.suggestions.style.display = 'block';
    }
    
    hideSuggestions() {
        this.suggestions.style.display = 'none';
    }
    
    selectSuggestion(suggestion) {
        this.messageInput.value = suggestion;
        this.hideSuggestions();
        this.messageInput.focus();
        
        // Trigger input event
        this.messageInput.dispatchEvent(new Event('input'));
    }
    
    // === AYUDA ===
    
    showHelp() {
        const helpMessage = `
🧮 **Guía del ChatBot Matemático con IA y Gráficas**

**📝 Cálculos:**
• Escribe directamente: "2+2", "sqrt(16)", "sin(pi/2)"
• Usa lenguaje natural: "cuánto es 5+3"

**📊 Gráficas:**
• "grafica sin(x)" - Función seno
• "grafica x^2" - Parábola
• "grafica 3x+2" - Función lineal
• "f(x) = x^2+1" - Notación matemática
• "compara sin(x) vs cos(x)" - Múltiples funciones

**🧠 Conceptos:**
• "qué es pi" - Explicaciones profundas
• "para qué sirven las derivadas" - Aplicaciones

**🔢 Herramientas:**
• Click en 🔢 para teclado matemático
• Click en ⚡ para ejemplos rápidos
• Click en 📊 para gráficas de ejemplo

**💡 Funciones soportadas:**
• Trigonométricas: sin, cos, tan
• Logarítmicas: log, exp
• Algebraicas: cualquier polinomio
• Constantes: pi, e`;

        this.addMessage(helpMessage, 'bot');
    }
    
    // === GRÁFICAS MEJORADAS ===
    
    createChart(chartData) {
        this.chartCount++;
        
        console.log('📊 Creando gráfica con datos:', chartData);
        
        // Crear contenedor para la gráfica
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        chartContainer.innerHTML = `
            <canvas id="chart-${this.chartCount}" width="400" height="300"></canvas>
        `;
        
        // Agregar al chat
        this.chatMessages.appendChild(chartContainer);
        this.scrollToBottom();
        
        // Crear la gráfica con Chart.js
        const ctx = document.getElementById(`chart-${this.chartCount}`).getContext('2d');
        
        try {
            new Chart(ctx, chartData);
            console.log('✅ Gráfica creada exitosamente');
        } catch (error) {
            console.error('❌ Error creando gráfica:', error);
            chartContainer.innerHTML = '<div class="chart-error">Error creando la gráfica: ' + error.message + '</div>';
        }
    }
    
    // === CONEXIÓN ===
    
    async checkConnection() {
        try {
            const response = await fetch(this.healthUrl, {
                method: 'GET',
                signal: AbortSignal.timeout(5000)
            });
            
            if (response.ok) {
                this.updateStatus('Conectado', 'connected');
            } else {
                this.updateStatus('Error de servidor', 'error');
            }
        } catch (error) {
            this.updateStatus('Sin conexión', 'disconnected');
            // Solo hacer console.log en desarrollo
            if (window.location.hostname === 'localhost') {
                console.error('Error de conexión:', error);
            }
        }
    }
    
    updateStatus(text, className) {
        this.status.textContent = text;
        this.status.className = `status ${className}`;
    }
    
    // === ENVÍO DE MENSAJES MEJORADO ===
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        
        if (!message) {
            this.messageInput.focus();
            return;
        }
        
        // Ocultar paneles y sugerencias
        this.hideAllPanels();
        this.hideSuggestions();
        
        // Ocultar panel de bienvenida si aún está visible
        if (!this.welcomePanelHidden) {
            this.hideWelcomePanelSmooth();
        }
        
        // Deshabilitar controles
        this.setInputsEnabled(false);
        
        // Mostrar mensaje del usuario
        this.addMessage(message, 'user');
        
        // Limpiar input
        this.messageInput.value = '';
        
        // Mostrar indicador de escritura
        this.showTypingIndicator();
        
        try {
            console.log('📤 Enviando mensaje:', message);
            
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
                signal: AbortSignal.timeout(25000) // 25 segundos para gráficas
            });
            
            this.hideTypingIndicator();
            
            if (response.ok) {
                const data = await response.json();
                console.log('📥 Respuesta recibida:', data);
                
                // Simular delay natural
                await this.delay(300);
                
                // Mostrar respuesta del bot
                this.addMessage(data.response || 'Lo siento, no pude generar una respuesta.', 'bot');
                
                // Si hay datos de gráfica, mostrarla
                if (data.chart_data) {
                    console.log('📊 Datos de gráfica detectados, creando visualización...');
                    await this.delay(500); // Pequeña pausa antes de la gráfica
                    this.createChart(data.chart_data);
                } else {
                    console.log('ℹ️ No se recibieron datos de gráfica');
                }
                
                this.updateStatus('Conectado', 'connected');
                
            } else {
                const errorData = await response.json().catch(() => ({}));
                this.addMessage('Lo siento, hubo un problema con el servidor. Inténtalo de nuevo.', 'bot');
                this.updateStatus('Error', 'error');
                
                // Solo hacer console.log en desarrollo
                if (window.location.hostname === 'localhost') {
                    console.error('Error del servidor:', errorData);
                }
            }
            
        } catch (error) {
            this.hideTypingIndicator();
            
            if (error.name === 'AbortError') {
                this.addMessage('La respuesta tardó demasiado. Inténtalo de nuevo.', 'bot');
            } else {
                this.addMessage('No pude conectar con el servidor. Verifica tu conexión.', 'bot');
            }
            
            this.updateStatus('Sin conexión', 'disconnected');
            
            // Solo hacer console.log en desarrollo
            if (window.location.hostname === 'localhost') {
                console.error('Error:', error);
            }
        }
        
        this.setInputsEnabled(true);
    }
    
    setInputsEnabled(enabled) {
        this.messageInput.disabled = !enabled;
        this.sendButton.disabled = !enabled;
        
        if (enabled) {
            this.messageInput.focus();
        }
    }
    
    addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Formatear texto si es del bot (soportar markdown básico)
        if (sender === 'bot') {
            contentDiv.innerHTML = this.formatBotMessage(text);
        } else {
            contentDiv.textContent = text;
        }
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = new Date().toLocaleTimeString('es-ES', {
            hour: '2-digit',
            minute: '2-digit'
        });
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    formatBotMessage(text) {
        // Formatear texto básico (bold, código)
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/`(.*?)`/g, '<code style="background:#f0f0f0;padding:2px 4px;border-radius:3px;font-family:monospace;">$1</code>')
            .replace(/\n/g, '<br>');
    }
    
    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message';
        typingDiv.id = 'typing-indicator';
        
        typingDiv.innerHTML = `
            <div class="message-content" style="display: flex; align-items: center; gap: 8px;">
                <span>Procesando</span>
                <div style="display: flex; gap: 4px;">
                    <span style="width: 8px; height: 8px; background: #ff0000; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out;"></span>
                    <span style="width: 8px; height: 8px; background: #ff0000; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out; animation-delay: -0.16s;"></span>
                    <span style="width: 8px; height: 8px; background: #ff0000; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out; animation-delay: -0.32s;"></span>
                </div>
            </div>
        `;
        
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// === FUNCIONES GLOBALES PARA EVENTOS ONCLICK ===

let chatBot;

function selectCategory(type) {
    chatBot.selectCategory(type);
}

function showExamples(category) {
    chatBot.showExamples(category);
}

function showCategories() {
    chatBot.showCategories();
}

function sendExample(example) {
    chatBot.messageInput.value = example;
    chatBot.hideAllPanels();
    
    // Trigger input event para ocultar welcome panel si es necesario
    chatBot.messageInput.dispatchEvent(new Event('input'));
    
    chatBot.sendMessage();
}

function insertSymbol(symbol) {
    chatBot.insertSymbol(symbol);
}

function insertFunction(func) {
    chatBot.insertFunction(func);
}

function clearInput() {
    chatBot.clearInput();
}

function toggleKeyboard() {
    chatBot.toggleKeyboard();
}

function toggleExamples() {
    chatBot.toggleExamples();
}

function toggleCharts() {
    chatBot.toggleCharts();
}

function showHelp() {
    chatBot.showHelp();
}

function selectSuggestion(suggestion) {
    chatBot.selectSuggestion(suggestion);
}

// Inicializar cuando se carga la página
document.addEventListener('DOMContentLoaded', () => {
    chatBot = new MathChatBot();
});