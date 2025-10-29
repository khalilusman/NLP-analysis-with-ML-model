// const API_URL = 'http://localhost:8000';
// let questions = [];
// let currentQuestionIndex = 0;
// let answers = {};
// let companyName = '';
// let isSubmitting = false;
// let assessmentComplete = false; // NEW: Flag to stop everything after results
// const chatContainer = document.getElementById('chatContainer');

// // Initialize on page load
// window.addEventListener('DOMContentLoaded', function() {
//     console.log("🎬 DOMContentLoaded event fired");
//     init();
// });

// // Prevent ANY page reloads
// window.addEventListener('beforeunload', function(e) {
//     if (assessmentComplete && !window.manualReload) {
//         console.log("⚠️ Someone tried to reload the page!");
//         e.preventDefault();
//         e.returnValue = '';
//         return "Results are displayed. Are you sure you want to leave?";
//     }
// });

// // Log any location changes
// const originalReload = window.location.reload;
// window.location.reload = function() {
//     console.log("🚨 window.location.reload() was called!");
//     console.trace(); // Show call stack
//     if (!window.manualReload) {
//         console.error("❌ UNAUTHORIZED RELOAD BLOCKED!");
//         return;
//     }
//     originalReload.call(window.location);
// };

// async function init() {
//     console.log("🎬 init() called - Initializing chatbot...");
//     console.log("Current state: assessmentComplete =", assessmentComplete);
    
//     // If assessment is already complete, DON'T reinitialize
//     if (assessmentComplete) {
//         console.log("🛑 Assessment already complete - BLOCKING re-initialization");
//         return;
//     }
    
//     try {
//         const response = await fetch(`${API_URL}/questions`);
//         const data = await response.json();
//         questions = data.questions;
        
//         console.log(`✅ Loaded ${questions.length} questions`);
        
//         addAssistantMessage("Hello! I'm your Risk Assessment Assistant.");
        
//         setTimeout(() => {
//             addAssistantMessage("I'll help you identify potential risks for your organization. Let's get started!");
//             setTimeout(() => askNextQuestion(), 1000);
//         }, 1500);
        
//     } catch (error) {
//         console.error('❌ Error loading questions:', error);
//         addAssistantMessage("I'm having trouble connecting to the server. Please ensure the backend is running on http://localhost:8000");
//     }
// }

// function addAssistantMessage(text) {
//     console.log("💬 addAssistantMessage called:", text.substring(0, 50));
//     const messageDiv = document.createElement('div');
//     messageDiv.className = 'message assistant';
//     messageDiv.innerHTML = `<div class="message-content">${text}</div>`;
//     chatContainer.appendChild(messageDiv);
//     scrollToBottom();
// }

// function addUserMessage(text) {
//     console.log("👤 addUserMessage called:", text.substring(0, 50));
//     const messageDiv = document.createElement('div');
//     messageDiv.className = 'message user';
//     messageDiv.innerHTML = `<div class="message-content">${text}</div>`;
//     chatContainer.appendChild(messageDiv);
//     scrollToBottom();
// }

// function showTyping() {
//     const typingDiv = document.createElement('div');
//     typingDiv.className = 'message assistant';
//     typingDiv.id = 'typing';
//     typingDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
//     chatContainer.appendChild(typingDiv);
//     scrollToBottom();
// }

// function hideTyping() {
//     const typing = document.getElementById('typing');
//     if (typing) typing.remove();
// }

// function scrollToBottom() {
//     chatContainer.scrollTop = chatContainer.scrollHeight;
// }

// function askNextQuestion() {
//     // CRITICAL: Stop if assessment is complete
//     if (assessmentComplete) {
//         console.log("🛑 ASSESSMENT COMPLETE - BLOCKING NEW QUESTIONS");
//         return;
//     }
    
//     console.log(`📝 Question ${currentQuestionIndex + 1}/${questions.length}`);
    
//     if (isSubmitting) {
//         console.log("⛔ Submission in progress, blocking");
//         return;
//     }
    
//     if (currentQuestionIndex >= questions.length) {
//         console.log("🎯 All questions answered, submitting...");
//         submitSurvey();
//         return;
//     }
    
//     const question = questions[currentQuestionIndex];
//     showTyping();
    
//     setTimeout(() => {
//         // DOUBLE CHECK before displaying
//         if (assessmentComplete) {
//             console.log("🛑 Assessment completed during timeout - aborting");
//             hideTyping();
//             return;
//         }
//         hideTyping();
//         displayQuestion(question);
//     }, 800);
// }

// function displayQuestion(question) {
//     const messageDiv = document.createElement('div');
//     messageDiv.className = 'message assistant';
    
//     const progress = ((currentQuestionIndex + 1) / questions.length) * 100;
//     let content = `<div class="message-content">${question.question}`;
//     content += `<div class="progress-bar"><div class="progress-fill" style="width: ${progress}%"></div></div>`;
//     content += `<div class="progress-text">Question ${currentQuestionIndex + 1} of ${questions.length}</div>`;

//     if (question.type === 'radio') {
//         content += '<div class="options-container">';
//         question.options.forEach(option => {
//             content += `<button class="option-btn" data-type="radio" data-key="${question.key}" data-value="${option}">${option}</button>`;
//         });
//         content += '</div>';
//     } else if (question.type === 'text') {
//         content += `<div class="text-input-container">
//             <input type="text" class="text-input" id="text-${question.key}" placeholder="Type your answer...">
//             <button class="submit-btn" data-type="text" data-key="${question.key}">Submit</button>
//         </div>`;
//     } else if (question.type === 'scale') {
//         content += '<div class="scale-container">';
//         for (let i = question.scale.min; i <= question.scale.max; i++) {
//             content += `<button class="scale-btn" data-type="scale" data-key="${question.key}" data-value="${i}">${i}</button>`;
//         }
//         content += '</div>';
//     } else if (question.type === 'scale_multiple') {
//         content += '<div class="scale-multiple-container">';
//         question.sub_questions.forEach(sq => {
//             content += `<div class="scale-question">
//                 <label class="scale-question-label">${sq.label}</label>
//                 <div class="scale-container" id="scale-${sq.key}">`;
//             for (let i = question.scale.min; i <= question.scale.max; i++) {
//                 content += `<button class="scale-btn" data-subkey="${sq.key}" data-value="${i}">${i}</button>`;
//             }
//             content += '</div></div>';
//         });
//         content += '</div>';
//         content += `<button class="submit-btn" data-type="scale_multiple" data-key="${question.key}">Continue</button>`;
//     }

//     content += '</div>';
//     messageDiv.innerHTML = content;
//     chatContainer.appendChild(messageDiv);
//     scrollToBottom();
    
//     // Attach event listeners
//     attachEventListeners(question);
// }

// function attachEventListeners(question) {
//     const buttons = chatContainer.querySelectorAll('button[data-type]');
    
//     buttons.forEach(btn => {
//         btn.addEventListener('click', function(e) {
//             e.preventDefault();
//             e.stopPropagation();
            
//             if (isSubmitting) return;
            
//             const type = this.getAttribute('data-type');
//             const key = this.getAttribute('data-key');
//             const value = this.getAttribute('data-value');
            
//             console.log(`🖱️ Clicked: ${type} - ${key} = ${value}`);
            
//             if (type === 'radio') {
//                 handleRadioAnswer(key, value);
//             } else if (type === 'text') {
//                 handleTextAnswer(key);
//             } else if (type === 'scale') {
//                 handleScaleAnswer(key, parseInt(value));
//             } else if (type === 'scale_multiple') {
//                 handleScaleMultipleAnswer(key, question.sub_questions);
//             }
//         });
//     });
    
//     // For scale_multiple, handle selection highlighting
//     if (question.type === 'scale_multiple') {
//         const scaleButtons = chatContainer.querySelectorAll('.scale-btn[data-subkey]');
//         scaleButtons.forEach(btn => {
//             btn.addEventListener('click', function(e) {
//                 e.stopPropagation();
//                 const subkey = this.getAttribute('data-subkey');
//                 const container = document.getElementById(`scale-${subkey}`);
//                 container.querySelectorAll('.scale-btn').forEach(b => b.classList.remove('selected'));
//                 this.classList.add('selected');
//             });
//         });
//     }
// }

// function handleRadioAnswer(key, value) {
//     if (assessmentComplete) {
//         console.log("🛑 Assessment complete - ignoring input");
//         return;
//     }
//     answers[key] = value;
//     if (key === 'company_name') companyName = value || 'Anonymous Company';
//     addUserMessage(value);
//     currentQuestionIndex++;
//     askNextQuestion();
// }

// function handleTextAnswer(key) {
//     if (assessmentComplete) {
//         console.log("🛑 Assessment complete - ignoring input");
//         return;
//     }
//     const input = document.getElementById(`text-${key}`);
//     const value = input.value.trim();
//     if (!value) {
//         alert('Please enter an answer');
//         return;
//     }
//     answers[key] = value;
//     if (key === 'company_name') companyName = value;
//     addUserMessage(value);
//     currentQuestionIndex++;
//     askNextQuestion();
// }

// function handleScaleAnswer(key, value) {
//     if (assessmentComplete) {
//         console.log("🛑 Assessment complete - ignoring input");
//         return;
//     }
//     answers[key] = value;
//     addUserMessage(`Selected: ${value}`);
//     currentQuestionIndex++;
//     askNextQuestion();
// }

// function handleScaleMultipleAnswer(key, subQuestions) {
//     if (assessmentComplete) {
//         console.log("🛑 Assessment complete - ignoring input");
//         return;
//     }
    
//     const scaleAnswers = {};
//     let allAnswered = true;

//     subQuestions.forEach(sq => {
//         const selectedBtn = document.querySelector(`#scale-${sq.key} .scale-btn.selected`);
//         if (selectedBtn) {
//             scaleAnswers[sq.key] = parseInt(selectedBtn.getAttribute('data-value'));
//         } else {
//             allAnswered = false;
//         }
//     });

//     if (!allAnswered) {
//         alert('Please rate all items');
//         return;
//     }

//     answers[key] = scaleAnswers;
//     const summary = subQuestions.map(sq => `${sq.label}: ${scaleAnswers[sq.key]}`).join(', ');
//     addUserMessage(summary);
//     currentQuestionIndex++;
//     askNextQuestion();
// }

// async function submitSurvey() {
//     if (isSubmitting) {
//         console.log("⛔ Already submitting");
//         return;
//     }
    
//     if (assessmentComplete) {
//         console.log("🛑 Assessment already complete");
//         return;
//     }
    
//     isSubmitting = true;
//     assessmentComplete = true; // LOCK IMMEDIATELY - NEVER ASK QUESTIONS AGAIN
//     console.log("=== 🚀 SUBMITTING SURVEY ===");
//     console.log("🔒 Assessment marked as COMPLETE");
    
//     // Disable all buttons permanently
//     document.querySelectorAll('button').forEach(btn => btn.disabled = true);
    
//     addAssistantMessage("Thank you for completing the assessment. Analyzing your responses...");
//     showTyping();
    
//     try {
//         console.log("📤 Sending to backend...");
        
//         const response = await fetch(`${API_URL}/predict`, {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify({ 
//                 company_name: companyName || "Anonymous Company", 
//                 answers: answers 
//             })
//         });
        
//         console.log(`📥 Response status: ${response.status}`);
        
//         if (!response.ok) {
//             const errorData = await response.json();
//             console.error("❌ Error:", errorData);
//             hideTyping();
//             addAssistantMessage(`Error: ${errorData.detail || 'Failed to get prediction'}`);
//             return;
//         }
        
//         const result = await response.json();
//         console.log("✅ Got result:", result);
        
//         hideTyping();
//         displayResults(result);
        
//     } catch (error) {
//         hideTyping();
//         console.error('❌ Fetch error:', error);
//         addAssistantMessage("An error occurred. Please try again.");
//     }
// }

// function displayResults(result) {
//     console.log("📊 DISPLAYING RESULTS - THIS IS THE END");
//     console.log("🛑 NO MORE QUESTIONS WILL BE ASKED");
    
//     const messageDiv = document.createElement('div');
//     messageDiv.className = 'message assistant';

//     const confidencePercent = result.confidence 
//         ? (result.confidence * 100).toFixed(1) 
//         : 'N/A';

//     let content = `
//         <div class="message-content">
//             <div class="result-card">
//                 <h3>✅ Risk Assessment Complete</h3>
//                 <p><strong>Predicted Risk:</strong> ${result.predicted_risk || 'Unknown'}</p>
//                 <br>
//                 <p><strong>💡 AI Recommendation:</strong></p>
//                 <p>${result.advice || 'No advice available'}</p>
//             </div>
//             <p style="margin-top: 20px; color: #8e8ea0; font-size: 14px;">
//                 Assessment complete. Refresh the page to start a new assessment.
//             </p>
//             <button class="submit-btn" id="restartBtn" style="background: #10a37f; margin-top: 16px;">
//                 🔄 Start New Assessment
//             </button>
//         </div>
//     `;

//     messageDiv.innerHTML = content;
//     chatContainer.appendChild(messageDiv);
//     scrollToBottom();
    
//     console.log("✅ RESULTS DISPLAYED");
//     console.log("⏹️ CHATBOT STOPPED - Session ended");
    
//     // ONLY restart when button is clicked
//     setTimeout(() => {
//         const restartBtn = document.getElementById('restartBtn');
//         if (restartBtn) {
//             restartBtn.disabled = false;
//             restartBtn.addEventListener('click', function(e) {
//                 e.preventDefault();
//                 e.stopPropagation();
//                 console.log("🔄 User manually clicked restart button");
//                 window.manualReload = true; // Flag to allow reload
//                 window.location.reload();
//             });
//         }
//     }, 100);
// }





























// const API_URL = 'http://127.0.0.1:8000';
// let questions = [];
// let currentQuestionIndex = 0;
// let answers = {};
// let companyName = '';
// let isSubmitting = false;
// let assessmentComplete = false;
// const chatContainer = document.getElementById('chatContainer');

// // Prevent unauthorized reloads
// window.addEventListener('beforeunload', function(e) {
//     if (assessmentComplete && !window.manualReload) {
//         e.preventDefault();
//         e.returnValue = '';
//         return "Results are displayed. Are you sure you want to leave?";
//     }
// });

// window.addEventListener('DOMContentLoaded', function() {
//     console.log("🎬 NLP Chatbot initializing...");
//     init();
// });

// async function init() {
//     console.log("🤖 Loading NLP-powered questions...");
    
//     if (assessmentComplete) {
//         console.log("🛑 Assessment already complete");
//         return;
//     }
    
//     try {
//         const response = await fetch(`${API_URL}/questions`);
//         const data = await response.json();
//         questions = data.questions;
        
//         console.log(`✅ Loaded ${questions.length} questions`);
        
//         addAssistantMessage("Hello! 👋 I'm your AI-powered Risk Assessment Assistant.");
        
//         setTimeout(() => {
//             addAssistantMessage("I'll have a conversation with you to understand your company's situation. Feel free to answer in your own words - I use Natural Language Processing to analyze your responses.");
//             setTimeout(() => askNextQuestion(), 1500);
//         }, 2000);
        
//     } catch (error) {
//         console.error('❌ Error loading questions:', error);
//         addAssistantMessage("I'm having trouble connecting. Please ensure the NLP backend is running on http://localhost:8000");
//     }
// }

// function addAssistantMessage(text) {
//     const messageDiv = document.createElement('div');
//     messageDiv.className = 'message assistant';
//     messageDiv.innerHTML = `<div class="message-content">${text}</div>`;
//     chatContainer.appendChild(messageDiv);
//     scrollToBottom();
// }

// function addUserMessage(text) {
//     const messageDiv = document.createElement('div');
//     messageDiv.className = 'message user';
//     messageDiv.innerHTML = `<div class="message-content">${text}</div>`;
//     chatContainer.appendChild(messageDiv);
//     scrollToBottom();
// }

// function showTyping() {
//     const typingDiv = document.createElement('div');
//     typingDiv.className = 'message assistant';
//     typingDiv.id = 'typing';
//     typingDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
//     chatContainer.appendChild(typingDiv);
//     scrollToBottom();
// }

// function hideTyping() {
//     const typing = document.getElementById('typing');
//     if (typing) typing.remove();
// }

// function scrollToBottom() {
//     chatContainer.scrollTop = chatContainer.scrollHeight;
// }

// function askNextQuestion() {
//     if (assessmentComplete) {
//         console.log("🛑 Assessment complete - blocking");
//         return;
//     }
    
//     console.log(`📝 Question ${currentQuestionIndex + 1}/${questions.length}`);
    
//     if (isSubmitting) {
//         console.log("⛔ Submission in progress");
//         return;
//     }
    
//     if (currentQuestionIndex >= questions.length) {
//         console.log("🎯 All questions answered, submitting to NLP backend...");
//         submitToNLP();
//         return;
//     }
    
//     const question = questions[currentQuestionIndex];
//     showTyping();
    
//     setTimeout(() => {
//         if (assessmentComplete) {
//             hideTyping();
//             return;
//         }
//         hideTyping();
//         displayQuestion(question);
//     }, 800);
// }

// function displayQuestion(question) {
//     const messageDiv = document.createElement('div');
//     messageDiv.className = 'message assistant';
    
//     const progress = ((currentQuestionIndex + 1) / questions.length) * 100;
//     let content = `<div class="message-content">${question.question}`;
//     content += `<div class="progress-bar"><div class="progress-fill" style="width: ${progress}%"></div></div>`;
//     content += `<div class="progress-text">Question ${currentQuestionIndex + 1} of ${questions.length}</div>`;

//     if (question.type === 'text' || question.type === 'text_long') {
//         const rows = question.type === 'text_long' ? 4 : 1;
//         const placeholder = question.placeholder || "Type your answer...";
        
//         content += `<div class="text-input-container">
//             <textarea 
//                 class="text-input text-area" 
//                 id="text-${question.key}" 
//                 rows="${rows}"
//                 placeholder="${placeholder}"
//                 style="resize: vertical; min-height: ${rows * 30}px;"
//             ></textarea>
//             <button class="submit-btn" data-type="text" data-key="${question.key}">Continue</button>
//         </div>`;
//     } else if (question.type === 'scale_multiple') {
//         content += '<div class="scale-multiple-container">';
//         question.sub_questions.forEach(sq => {
//             content += `<div class="scale-question">
//                 <label class="scale-question-label">${sq.label}</label>
//                 <div class="scale-container" id="scale-${sq.key}">`;
//             for (let i = question.scale.min; i <= question.scale.max; i++) {
//                 content += `<button class="scale-btn" data-subkey="${sq.key}" data-value="${i}">${i}</button>`;
//             }
//             content += '</div></div>';
//         });
//         content += '</div>';
//         content += `<button class="submit-btn" data-type="scale_multiple" data-key="${question.key}">Continue</button>`;
//     }

//     content += '</div>';
//     messageDiv.innerHTML = content;
//     chatContainer.appendChild(messageDiv);
//     scrollToBottom();
    
//     attachEventListeners(question);
// }

// function attachEventListeners(question) {
//     const buttons = chatContainer.querySelectorAll('button[data-type]');
    
//     buttons.forEach(btn => {
//         btn.addEventListener('click', function(e) {
//             e.preventDefault();
//             e.stopPropagation();
            
//             if (assessmentComplete) return;
            
//             const type = this.getAttribute('data-type');
//             const key = this.getAttribute('data-key');
            
//             if (type === 'text') {
//                 handleTextAnswer(key);
//             } else if (type === 'scale_multiple') {
//                 handleScaleMultipleAnswer(key, question.sub_questions);
//             }
//         });
//     });
    
//     if (question.type === 'scale_multiple') {
//         const scaleButtons = chatContainer.querySelectorAll('.scale-btn[data-subkey]');
//         scaleButtons.forEach(btn => {
//             btn.addEventListener('click', function(e) {
//                 e.stopPropagation();
//                 const subkey = this.getAttribute('data-subkey');
//                 const container = document.getElementById(`scale-${subkey}`);
//                 container.querySelectorAll('.scale-btn').forEach(b => b.classList.remove('selected'));
//                 this.classList.add('selected');
//             });
//         });
//     }
// }

// function handleTextAnswer(key) {
//     if (assessmentComplete) return;
    
//     const input = document.getElementById(`text-${key}`);
//     const value = input.value.trim();
    
//     if (!value) {
//         alert('Please enter an answer');
//         return;
//     }
    
//     answers[key] = value;
//     if (key === 'company_name') companyName = value;
    
//     // Show shortened version in chat
//     const displayText = value.length > 100 ? value.substring(0, 100) + "..." : value;
//     addUserMessage(displayText);
    
//     currentQuestionIndex++;
//     askNextQuestion();
// }

// function handleScaleMultipleAnswer(key, subQuestions) {
//     if (assessmentComplete) return;
    
//     const scaleAnswers = {};
//     let allAnswered = true;

//     subQuestions.forEach(sq => {
//         const selectedBtn = document.querySelector(`#scale-${sq.key} .scale-btn.selected`);
//         if (selectedBtn) {
//             scaleAnswers[sq.key] = parseInt(selectedBtn.getAttribute('data-value'));
//         } else {
//             allAnswered = false;
//         }
//     });

//     if (!allAnswered) {
//         alert('Please rate all items');
//         return;
//     }

//     answers[key] = scaleAnswers;
//     const summary = subQuestions.map(sq => `${sq.label}: ${scaleAnswers[sq.key]}`).join(', ');
//     addUserMessage(summary);
    
//     currentQuestionIndex++;
//     askNextQuestion();
// }

// async function submitToNLP() {
//     if (isSubmitting || assessmentComplete) {
//         console.log("⛔ Already submitted");
//         return;
//     }
    
//     isSubmitting = true;
//     assessmentComplete = true;
//     console.log("=== 🚀 SUBMITTING TO NLP BACKEND ===");
    
//     document.querySelectorAll('button').forEach(btn => btn.disabled = true);
    
//     addAssistantMessage("Thank you! 🤖 I'm now analyzing your responses using Natural Language Processing...");
//     showTyping();
    
//     try {
//         // Separate text and structured responses
//         const textResponses = {};
//         const structuredData = {};
        
//         for (const [key, value] of Object.entries(answers)) {
//             if (typeof value === 'string' && value.length > 20) {
//                 textResponses[key] = value;
//             } else {
//                 structuredData[key] = value;
//             }
//         }
        
//         console.log("📤 Text responses:", Object.keys(textResponses).length);
//         console.log("📊 Structured data:", Object.keys(structuredData).length);
        
//         const response = await fetch(`${API_URL}/predict-nlp`, {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify({
//                 company_name: companyName || "Anonymous Company",
//                 text_responses: textResponses,
//                 structured_data: structuredData
//             })
//         });
        
//         console.log(`📥 Response status: ${response.status}`);
        
//         if (!response.ok) {
//             const errorData = await response.json();
//             console.error("❌ Error:", errorData);
//             hideTyping();
//             addAssistantMessage(`Error: ${errorData.detail || 'NLP analysis failed'}`);
//             return;
//         }
        
//         const result = await response.json();
//         console.log("✅ NLP Analysis complete:", result);
        
//         hideTyping();
//         displayNLPResults(result);
        
//     } catch (error) {
//         hideTyping();
//         console.error('❌ Fetch error:', error);
//         addAssistantMessage("An error occurred during NLP analysis. Please try again.");
//     }
// }

// function displayNLPResults(result) {
//     console.log("📊 DISPLAYING NLP RESULTS");
    
//     const messageDiv = document.createElement('div');
//     messageDiv.className = 'message assistant';

//     const confidencePercent = result.confidence 
//         ? (result.confidence * 100).toFixed(1) 
//         : 'N/A';
    
//     const sentiment = result.sentiment_analysis;
//     const sentimentLabel = sentiment?.label || 'N/A';
//     const sentimentEmoji = sentimentLabel === 'POSITIVE' ? '😊' : sentimentLabel === 'NEGATIVE' ? '😟' : '😐';

//     let content = `
//         <div class="message-content">
//             <div class="result-card">
//                 <h3>🤖 NLP Analysis Complete</h3>
                
//                 <p><strong>🎯 Predicted Risk:</strong> ${result.predicted_risk || 'Unknown'}</p>
//                 <p><strong>📊 Confidence:</strong> ${confidencePercent}%</p>
                
//                 ${result.confidence ? `
//                 <div class="confidence-bar">
//                     <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
//                 </div>
//                 ` : ''}
                
//                 <br>
                
//                 <p><strong>${sentimentEmoji} Sentiment Analysis:</strong> ${sentimentLabel}</p>
//                 ${sentiment?.polarity ? `<p style="font-size: 14px; color: #8e8ea0;">Polarity: ${sentiment.polarity.toFixed(2)} (${sentiment.polarity > 0 ? 'Optimistic' : 'Concerned'})</p>` : ''}
                
//                 ${result.risk_keywords && result.risk_keywords.length > 0 ? `
//                 <br>
//                 <p><strong>🔑 Key Topics Detected:</strong></p>
//                 <p style="font-size: 14px; color: #c5c5d2;">${result.risk_keywords.slice(0, 8).join(', ')}</p>
//                 ` : ''}
                
//                 <br>
//                 <p><strong>💡 AI Recommendation:</strong></p>
//                 <p>${result.advice || 'No advice available'}</p>
//             </div>
            
//             <p style="margin-top: 20px; color: #8e8ea0; font-size: 14px; font-style: italic;">
//                 ✨ This analysis was powered by Natural Language Processing, including sentiment analysis, keyword extraction, and zero-shot risk classification.
//             </p>
            
//             <button class="submit-btn" id="restartBtn" style="background: #10a37f; margin-top: 16px;">
//                 🔄 Start New Assessment
//             </button>
//         </div>
//     `;

//     messageDiv.innerHTML = content;
//     chatContainer.appendChild(messageDiv);
//     scrollToBottom();
    
//     console.log("✅ NLP RESULTS DISPLAYED");
//     console.log("⏹️ CHATBOT STOPPED");
    
//     setTimeout(() => {
//         const restartBtn = document.getElementById('restartBtn');
//         if (restartBtn) {
//             restartBtn.disabled = false;
//             restartBtn.addEventListener('click', function(e) {
//                 e.preventDefault();
//                 console.log("🔄 User clicked restart");
//                 window.manualReload = true;
//                 window.location.reload();
//             });
//         }
//     }, 100);
// }

































const API_URL = 'http://localhost:8000';
let questions = [];
let currentQuestionIndex = 0;
let answers = {};
let customRisks = []; // NEW: Store custom risks
let companyName = '';
let isSubmitting = false;
let assessmentComplete = false;
const chatContainer = document.getElementById('chatContainer');

// Initialize on page load
window.addEventListener('DOMContentLoaded', function() {
    console.log("🎬 DOMContentLoaded event fired");
    init();
});

// Prevent ANY page reloads
window.addEventListener('beforeunload', function(e) {
    if (assessmentComplete && !window.manualReload) {
        console.log("⚠️ Someone tried to reload the page!");
        e.preventDefault();
        e.returnValue = '';
        return "Results are displayed. Are you sure you want to leave?";
    }
});

// Log any location changes
const originalReload = window.location.reload;
window.location.reload = function() {
    console.log("🚨 window.location.reload() was called!");
    console.trace();
    if (!window.manualReload) {
        console.error("❌ UNAUTHORIZED RELOAD BLOCKED!");
        return;
    }
    originalReload.call(window.location);
};

async function init() {
    console.log("🎬 init() called - Initializing chatbot...");
    console.log("Current state: assessmentComplete =", assessmentComplete);
    
    if (assessmentComplete) {
        console.log("🛑 Assessment already complete - BLOCKING re-initialization");
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/questions`);
        const data = await response.json();
        questions = data.questions;
        
        console.log(`✅ Loaded ${questions.length} questions`);
        
        addAssistantMessage("Hello! I'm your Risk Assessment Assistant.");
        
        setTimeout(() => {
            addAssistantMessage("I'll help you identify potential risks for your organization. Let's get started!");
            setTimeout(() => askNextQuestion(), 1000);
        }, 1500);
        
    } catch (error) {
        console.error('❌ Error loading questions:', error);
        addAssistantMessage("I'm having trouble connecting to the server. Please ensure the backend is running on http://localhost:8000");
    }
}

function addAssistantMessage(text) {
    console.log("💬 addAssistantMessage called:", text.substring(0, 50));
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `<div class="message-content">${text}</div>`;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function addUserMessage(text) {
    console.log("👤 addUserMessage called:", text.substring(0, 50));
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `<div class="message-content">${text}</div>`;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function showTyping() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = 'typing';
    typingDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    chatContainer.appendChild(typingDiv);
    scrollToBottom();
}

function hideTyping() {
    const typing = document.getElementById('typing');
    if (typing) typing.remove();
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function askNextQuestion() {
    if (assessmentComplete) {
        console.log("🛑 ASSESSMENT COMPLETE - BLOCKING NEW QUESTIONS");
        return;
    }
    
    console.log(`📝 Question ${currentQuestionIndex + 1}/${questions.length}`);
    
    if (isSubmitting) {
        console.log("⛔ Submission in progress, blocking");
        return;
    }
    
    if (currentQuestionIndex >= questions.length) {
        console.log("🎯 All questions answered, submitting...");
        submitSurvey();
        return;
    }
    
    const question = questions[currentQuestionIndex];
    showTyping();
    
    setTimeout(() => {
        if (assessmentComplete) {
            console.log("🛑 Assessment completed during timeout - aborting");
            hideTyping();
            return;
        }
        hideTyping();
        displayQuestion(question);
    }, 800);
}

function displayQuestion(question) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    const progress = ((currentQuestionIndex + 1) / questions.length) * 100;
    let content = `<div class="message-content">${question.question}`;
    content += `<div class="progress-bar"><div class="progress-fill" style="width: ${progress}%"></div></div>`;
    content += `<div class="progress-text">Question ${currentQuestionIndex + 1} of ${questions.length}</div>`;

    if (question.type === 'radio') {
        content += '<div class="options-container">';
        question.options.forEach(option => {
            content += `<button class="option-btn" data-type="radio" data-key="${question.key}" data-value="${option}">${option}</button>`;
        });
        content += '</div>';
    } else if (question.type === 'text') {
        content += `<div class="text-input-container">
            <input type="text" class="text-input" id="text-${question.key}" placeholder="Type your answer...">
            <button class="submit-btn" data-type="text" data-key="${question.key}">Submit</button>
        </div>`;
    } else if (question.type === 'scale') {
        content += '<div class="scale-container">';
        for (let i = question.scale.min; i <= question.scale.max; i++) {
            content += `<button class="scale-btn" data-type="scale" data-key="${question.key}" data-value="${i}">${i}</button>`;
        }
        content += '</div>';
    } else if (question.type === 'scale_multiple') {
        content += '<div class="scale-multiple-container">';
        question.sub_questions.forEach(sq => {
            content += `<div class="scale-question">
                <label class="scale-question-label">${sq.label}</label>
                <div class="scale-container" id="scale-${sq.key}">`;
            for (let i = question.scale.min; i <= question.scale.max; i++) {
                content += `<button class="scale-btn" data-subkey="${sq.key}" data-value="${i}">${i}</button>`;
            }
            content += '</div></div>';
        });
        
        // NEW: Add custom risks section
        content += '<div id="custom-risks-section" style="margin-top: 25px; padding-top: 20px; border-top: 2px dashed #e5e7eb;">';
        content += '<p style="font-size: 14px; color: #6b7280; margin-bottom: 12px;">📝 Have other risks not listed above? Add them here:</p>';
        content += '<div id="custom-risks-list"></div>';
        content += '<button class="option-btn" id="add-custom-risk-btn" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin-top: 10px; border: none;">➕ Add Custom Risk</button>';
        content += '</div>';
        
        content += '</div>';
        content += `<button class="submit-btn" data-type="scale_multiple" data-key="${question.key}" style="margin-top: 20px;">Continue</button>`;
    }

    content += '</div>';
    messageDiv.innerHTML = content;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
    
    attachEventListeners(question);
}

function attachEventListeners(question) {
    const buttons = chatContainer.querySelectorAll('button[data-type]');
    
    buttons.forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            if (isSubmitting) return;
            
            const type = this.getAttribute('data-type');
            const key = this.getAttribute('data-key');
            const value = this.getAttribute('data-value');
            
            console.log(`🖱️ Clicked: ${type} - ${key} = ${value}`);
            
            if (type === 'radio') {
                handleRadioAnswer(key, value);
            } else if (type === 'text') {
                handleTextAnswer(key);
            } else if (type === 'scale') {
                handleScaleAnswer(key, parseInt(value));
            } else if (type === 'scale_multiple') {
                handleScaleMultipleAnswer(key, question.sub_questions);
            }
        });
    });
    
    // For scale_multiple, handle selection highlighting
    if (question.type === 'scale_multiple') {
        const scaleButtons = chatContainer.querySelectorAll('.scale-btn[data-subkey]');
        scaleButtons.forEach(btn => {
            btn.addEventListener('click', function(e) {
                e.stopPropagation();
                const subkey = this.getAttribute('data-subkey');
                const container = document.getElementById(`scale-${subkey}`);
                container.querySelectorAll('.scale-btn').forEach(b => b.classList.remove('selected'));
                this.classList.add('selected');
            });
        });
        
        // NEW: Handle "Add Custom Risk" button
        const addCustomRiskBtn = document.getElementById('add-custom-risk-btn');
        if (addCustomRiskBtn) {
            addCustomRiskBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                addCustomRiskInput();
            });
        }
    }
}

// NEW: Add custom risk input field
function addCustomRiskInput() {
    const customRisksList = document.getElementById('custom-risks-list');
    const riskId = `custom-risk-${Date.now()}`;
    
    const riskDiv = document.createElement('div');
    riskDiv.className = 'custom-risk-item';
    riskDiv.id = riskId;
    riskDiv.innerHTML = `
        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 12px; animation: slideIn 0.3s ease-out;">
            <input type="text" 
                   class="text-input custom-risk-input" 
                   id="input-${riskId}" 
                   placeholder="e.g., Data breach, Supply chain issues..."
                   style="flex: 1; padding: 12px; font-size: 14px; border: 2px solid #d1d5db; border-radius: 8px;">
            <button class="option-btn" onclick="saveCustomRisk('${riskId}')" style="background: #10a37f; padding: 10px 18px; border: none; min-width: 70px;">✓ Save</button>
            <button class="option-btn" onclick="removeCustomRiskInput('${riskId}')" style="background: #ef4444; padding: 10px 18px; border: none; min-width: 70px;">✗ Cancel</button>
        </div>
    `;
    
    customRisksList.appendChild(riskDiv);
    scrollToBottom();
    
    // Focus on the input
    setTimeout(() => {
        document.getElementById(`input-${riskId}`).focus();
    }, 100);
}

// NEW: Save custom risk
function saveCustomRisk(riskId) {
    const input = document.getElementById(`input-${riskId}`);
    const riskName = input.value.trim();
    
    if (!riskName) {
        alert('Please enter a risk name');
        return;
    }
    
    // Check for duplicates
    if (customRisks.includes(riskName)) {
        alert('This risk has already been added');
        return;
    }
    
    customRisks.push(riskName);
    console.log(`✅ Added custom risk: ${riskName}`);
    
    // Replace input with a label
    const riskDiv = document.getElementById(riskId);
    riskDiv.innerHTML = `
        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px; padding: 12px; background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%); border-radius: 8px; border-left: 4px solid #667eea;">
            <span style="flex: 1; font-weight: 500; color: #4c1d95;">✓ ${riskName}</span>
            <button class="option-btn" onclick="removeCustomRisk('${riskId}', '${riskName.replace(/'/g, "\\'")}' )" style="background: #ef4444; padding: 6px 14px; font-size: 13px; border: none;">Remove</button>
        </div>
    `;
}

// NEW: Remove custom risk input (before saving)
function removeCustomRiskInput(riskId) {
    const riskDiv = document.getElementById(riskId);
    if (riskDiv) {
        riskDiv.remove();
    }
}

// NEW: Remove saved custom risk
function removeCustomRisk(riskId, riskName) {
    customRisks = customRisks.filter(r => r !== riskName);
    console.log(`❌ Removed custom risk: ${riskName}`);
    
    const riskDiv = document.getElementById(riskId);
    if (riskDiv) {
        riskDiv.remove();
    }
}

// Make functions global
window.saveCustomRisk = saveCustomRisk;
window.removeCustomRiskInput = removeCustomRiskInput;
window.removeCustomRisk = removeCustomRisk;

function handleRadioAnswer(key, value) {
    if (assessmentComplete) {
        console.log("🛑 Assessment complete - ignoring input");
        return;
    }
    answers[key] = value;
    if (key === 'company_name') companyName = value || 'Anonymous Company';
    addUserMessage(value);
    currentQuestionIndex++;
    askNextQuestion();
}

function handleTextAnswer(key) {
    if (assessmentComplete) {
        console.log("🛑 Assessment complete - ignoring input");
        return;
    }
    const input = document.getElementById(`text-${key}`);
    const value = input.value.trim();
    if (!value) {
        alert('Please enter an answer');
        return;
    }
    answers[key] = value;
    if (key === 'company_name') companyName = value;
    addUserMessage(value);
    currentQuestionIndex++;
    askNextQuestion();
}

function handleScaleAnswer(key, value) {
    if (assessmentComplete) {
        console.log("🛑 Assessment complete - ignoring input");
        return;
    }
    answers[key] = value;
    addUserMessage(`Selected: ${value}`);
    currentQuestionIndex++;
    askNextQuestion();
}

function handleScaleMultipleAnswer(key, subQuestions) {
    if (assessmentComplete) {
        console.log("🛑 Assessment complete - ignoring input");
        return;
    }
    
    const scaleAnswers = {};
    let allAnswered = true;

    subQuestions.forEach(sq => {
        const selectedBtn = document.querySelector(`#scale-${sq.key} .scale-btn.selected`);
        if (selectedBtn) {
            scaleAnswers[sq.key] = parseInt(selectedBtn.getAttribute('data-value'));
        } else {
            allAnswered = false;
        }
    });

    if (!allAnswered) {
        alert('Please rate all items');
        return;
    }

    answers[key] = scaleAnswers;
    
    // Add custom risks summary to user message
    const summary = subQuestions.map(sq => `${sq.label}: ${scaleAnswers[sq.key]}`).join(', ');
    let userMessage = summary;
    
    if (customRisks.length > 0) {
        userMessage += `\n\n📝 Custom risks added: ${customRisks.join(', ')}`;
    }
    
    addUserMessage(userMessage);
    console.log(`📊 Custom risks: ${customRisks.length}`, customRisks);
    
    currentQuestionIndex++;
    askNextQuestion();
}

async function submitSurvey() {
    if (isSubmitting) {
        console.log("⛔ Already submitting");
        return;
    }
    
    if (assessmentComplete) {
        console.log("🛑 Assessment already complete");
        return;
    }
    
    isSubmitting = true;
    assessmentComplete = true;
    console.log("=== 🚀 SUBMITTING SURVEY ===");
    console.log("🔒 Assessment marked as COMPLETE");
    console.log(`📊 Submitting with ${customRisks.length} custom risks:`, customRisks);
    
    document.querySelectorAll('button').forEach(btn => btn.disabled = true);
    
    addAssistantMessage("Thank you for completing the assessment. Analyzing your responses...");
    showTyping();
    
    try {
        console.log("📤 Sending to backend...");
        
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                company_name: companyName || "Anonymous Company", 
                answers: answers,
                custom_risks: customRisks // NEW: Send custom risks
            })
        });
        
        console.log(`📥 Response status: ${response.status}`);
        
        if (!response.ok) {
            const errorData = await response.json();
            console.error("❌ Error:", errorData);
            hideTyping();
            addAssistantMessage(`Error: ${errorData.detail || 'Failed to get prediction'}`);
            return;
        }
        
        const result = await response.json();
        console.log("✅ Got result:", result);
        
        hideTyping();
        displayResults(result);
        
    } catch (error) {
        hideTyping();
        console.error('❌ Fetch error:', error);
        addAssistantMessage("An error occurred. Please try again.");
    }
}

function displayResults(result) {
    console.log("📊 DISPLAYING RESULTS - THIS IS THE END");
    console.log("🛑 NO MORE QUESTIONS WILL BE ASKED");
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';

    let content = `
        <div class="message-content">
            <div class="result-card-main">
                <h3 style="margin: 0 0 15px 0; font-size: 22px; display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 28px;">✅</span> Risk Assessment Complete
                </h3>
                <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">Predicted Risk Category</p>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: 700;">${result.predicted_risk || 'Unknown'}</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                    <p style="margin: 0 0 8px 0; font-weight: 600; font-size: 16px;">💡 AI Recommendation</p>
                    <p style="margin: 0; line-height: 1.7; font-size: 15px;">${result.advice || 'No advice available'}</p>
                </div>
            </div>
    `;
    
    // NEW: Display custom risk advice if available
    if (result.custom_risk_advice && result.custom_risk_advice.length > 0) {
        content += '<div class="result-card-custom">';
        content += '<h3 style="margin: 0 0 20px 0; font-size: 20px; display: flex; align-items: center; gap: 10px;"><span style="font-size: 26px;">🎯</span> Custom Risk Analysis</h3>';
        
        result.custom_risk_advice.forEach((item, index) => {
            content += `
                <div class="custom-risk-advice-item">
                    <div class="risk-number">${index + 1}</div>
                    <div style="flex: 1;">
                        <p class="risk-name">${item.risk_name}</p>
                        <p class="risk-advice">${item.advice}</p>
                    </div>
                </div>
            `;
        });
        
        content += '</div>';
    }
    
    content += `
            <p style="margin-top: 24px; color: #8e8ea0; font-size: 14px; text-align: center;">
                Assessment complete. Click below to start a new assessment.
            </p>
            <button class="submit-btn" id="restartBtn" style="background: linear-gradient(135deg, #10a37f 0%, #059669 100%); margin-top: 16px; width: 100%; font-size: 16px; padding: 14px; border: none;">
                🔄 Start New Assessment
            </button>
        </div>
    `;

    messageDiv.innerHTML = content;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
    
    console.log("✅ RESULTS DISPLAYED");
    console.log("⏹️ CHATBOT STOPPED - Session ended");
    
    setTimeout(() => {
        const restartBtn = document.getElementById('restartBtn');
        if (restartBtn) {
            restartBtn.disabled = false;
            restartBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log("🔄 User manually clicked restart button");
                window.manualReload = true;
                window.location.reload();
            });
        }
    }, 100);
}