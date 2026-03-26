class SITNAGPURChatbot {
    constructor() {
        this.context = { lastIntent: null, lastEntities: {}, turnCount: 0 };
this.analytics = {
    totalQueries: 0, successfulResponses: 0,
    fallbackCount: 0, handoverCount: 0,
    sessionStart: new Date(), popularTopics: {},
    queryLog: [], unansweredQueries: [], suggestedFAQs: {}
};
        this.messagesArea = document.getElementById('messagesArea');
        this.userInput = document.getElementById('userInput');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');
        
        // SITNAGPUR Knowledge Base
        this.knowledgeBase = {
            admissions: {
                title: "Admissions 2026",
                response: `**Admissions Open for 2026-27 Academic Year** 🎓

**Eligibility Criteria:**
• B.Tech: 60% in 10+2 with PCM
• M.Tech: 60% in B.Tech + GATE score
• MBA: 60% in graduation + CAT/MAT score
• PhD: Master's degree with 60% + entrance test

**Important Dates:**
• Application Start: January 15, 2026
• Last Date: March 31, 2026
• Entrance Test: April 15, 2026
• Result Declaration: May 10, 2026

**Required Documents:**
• 10th & 12th mark sheets
• Graduation certificates
• Entrance exam score card
• ID proof
• Passport size photographs

**Application Fee:**
• General: ₹1200
• SC/ST: ₹600

Would you like to know about specific programs or scholarship opportunities?`
            },
            
            courses: {
                title: "Courses Offered",
                response: `**Academic Programs at SITNAGPUR** 📚

**Undergraduate Programs (B.Tech):**
• Computer Science & Engineering
• Electronics & Communication
• Mechanical Engineering
• Civil Engineering
• Electrical Engineering
• Artificial Intelligence & ML
• Data Science
• Robotics & Automation

**Postgraduate Programs (M.Tech):**
• AI & Machine Learning
• VLSI Design
• Structural Engineering
• Thermal Engineering
• Computer Science

**Doctoral Programs (PhD):**
• Engineering & Technology
• Management Studies
• Applied Sciences

**Management Programs:**
• MBA in Technology Management
• MBA in Business Analytics

Each program offers specialization options and industry-integrated curriculum. Which program interests you?`
            },
            
            fees: {
                title: "Fee Structure",
                response: `**Fee Structure for Academic Year 2026-27** 💰

**Tuition Fees (Per Year):**
• B.Tech: ₹1,25,000
• M.Tech: ₹1,00,000
• MBA: ₹1,50,000
• PhD: ₹75,000

**Additional Fees:**
• Hostel (Optional): ₹60,000 - ₹85,000/year
• Mess: ₹36,000/year
• Library: ₹5,000/year
• Laboratory: ₹8,000/year
• Sports: ₹2,000/year

**One-time Fees:**
• Admission Fee: ₹10,000
• Caution Deposit (Refundable): ₹5,000
• Alumni Fee: ₹2,000

**Scholarship Opportunities:**
• Merit-based: Up to 100% tuition fee waiver
• Need-based: Up to 50% concession
• Sports quota: Up to 75% concession
• Girls scholarship: 10% additional discount
• SC/ST scholarships as per government norms

Need help calculating your total expenses or learning about payment plans?`
            },
            
            exams: {
                title: "Examination Schedule",
                response: `**Examination Schedule 2026** 📝

**Mid-Term Examinations:**
• Even Semester: March 10-20, 2026
• Odd Semester: September 15-25, 2026

**Final Examinations:**
• Even Semester: May 5-25, 2026
• Odd Semester: November 20 - December 10, 2026

**Practical/Viva Examinations:**
• Even Semester: April 15-30, 2026
• Odd Semester: November 5-15, 2026

**Supplementary Examinations:**
• Even Semester: July 10-20, 2026
• Odd Semester: January 10-20, 2027

**Important Guidelines:**
• Minimum 75% attendance required
• Hall tickets released 7 days before exams
• Results declared within 15 days
• Re-evaluation window: 7 days after results

Would you like to check your exam timetable or know more about exam rules?`
            },
            
            placement: {
                title: "Placement Statistics",
                response: `**Placement Highlights 2025-26** 💼

**Overall Statistics:**
• Placement Rate: 92%
• Average Package: ₹8.5 LPA
• Highest Package: ₹32 LPA
• Top 10% Average: ₹18 LPA
• Total Offers: 450+

**Top Recruiters:**
• Microsoft - 12 offers (₹32 LPA)
• Google - 8 offers (₹28 LPA)
• Amazon - 25 offers (₹24 LPA)
• Goldman Sachs - 15 offers (₹20 LPA)
• Tata Motors - 30 offers (₹12 LPA)
• Infosys - 45 offers (₹8 LPA)
• TCS - 60 offers (₹7.5 LPA)
• L&T - 35 offers (₹9 LPA)

**Internship Opportunities:**
• 85% students got internships
• Average stipend: ₹25,000/month
• International internships: 25 students

**Upcoming Drives (March 2026):**
• Microsoft (7th March)
• Amazon (10th March)
• Google (15th March)
• Intel (20th March)

Need help with placement preparation or want to see company-wise details?`
            },
            
            contact: {
                title: "Contact Information",
                response: `**SITNAGPUR Contact Details** 📞

**Main Campus:**
📍 Address: SIT Nagpur, Maharashtra - 440001
📞 Phone: +91 712 280 1234
📧 Email: info@sitnagpur.edu.in

**Admissions Office:**
📞 Direct: +91 712 280 5678
📧 Email: admissions@sitnagpur.edu.in
🕒 Hours: Mon-Fri, 9 AM - 5 PM

**Academic Office:**
📞 Phone: +91 712 280 9101
📧 Email: academics@sitnagpur.edu.in

**Placement Cell:**
📞 Phone: +91 712 280 1122
📧 Email: placement@sitnagpur.edu.in

**International Relations:**
📞 Phone: +91 712 280 3344
📧 Email: international@sitnagpur.edu.in

**Emergency Contact:**
🚑 24/7 Helpline: +91 712 280 9999

**Social Media:**
• Instagram: @sitnagpur_official
• LinkedIn: SIT Nagpur
• Twitter: @SITNagpur
• Facebook: /SITNagpur

Would you like directions to campus or want to schedule a visit?`
            },
            
            facilities: {
                title: "Campus Facilities",
                response: `**SITNAGPUR Campus Facilities** 🏛️

**Academic Facilities:**
• Smart Classrooms with IoT
• 25+ Advanced Laboratories
• Central Library with 50,000+ books
• Digital Learning Center
• Research & Innovation Hub

**Hostel Facilities:**
• Separate hostels for boys & girls
• Wi-Fi enabled campus
• 24/7 power backup
• Gymnasium & Fitness Center
• Common rooms with TV
• Laundry services

**Sports Complex:**
• Olympic size swimming pool
• Basketball & Tennis courts
• Football ground
• Indoor badminton courts
• Table tennis & chess rooms

**Other Amenities:**
• Medical center with ambulance
• Cafeteria & Food court
• Banking & ATM services
• Stationery store
• Transportation services
• 24/7 security with CCTV

**Upcoming Facilities:**
• Innovation Incubation Center
• E-Sports Arena
• Virtual Reality Lab

Want to know more about any specific facility?`
            }
        };

        this.init();
    }

    init() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });

        // Quick action buttons
        document.querySelectorAll('.quick-action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const query = e.currentTarget.dataset.query;
                this.handleQuickAction(query, e);
            });
        });

        // Chat list items
        document.querySelectorAll('.chat-list-item').forEach(item => {
            item.addEventListener('click', function() {
                document.querySelectorAll('.chat-list-item').forEach(i => i.classList.remove('active'));
                this.classList.add('active');
            });
        });
    }

    sendMessage() {
        const message = this.userInput.value.trim();
        if (message === '') return;

        // Add user message
        this.addMessage(message, 'user');
        this.userInput.value = '';

        // Show typing indicator
        this.showTypingIndicator();

        // Process and respond
        setTimeout(() => {
            this.hideTypingIndicator();
            this.processMessage(message);
        }, 1500);
    }

    addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        // Add avatar
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = sender === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
        messageDiv.appendChild(avatarDiv);

        // Message content wrapper
        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'message-content-wrapper';

        // Sender name
        const senderDiv = document.createElement('div');
        senderDiv.className = 'message-sender';
        senderDiv.textContent = sender === 'bot' ? 'SITNAGPUR Assistant' : 'You';
        contentWrapper.appendChild(senderDiv);

        // Message bubble
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        
        // Format message (handle markdown-like bold)
        const formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        bubbleDiv.innerHTML = formattedText.replace(/\n/g, '<br>');
        contentWrapper.appendChild(bubbleDiv);

        // Message time
        const timeSpan = document.createElement('span');
        timeSpan.className = 'message-time';
        timeSpan.textContent = this.getCurrentTime();
        contentWrapper.appendChild(timeSpan);

        messageDiv.appendChild(contentWrapper);
        this.messagesArea.appendChild(messageDiv);
        
        this.scrollToBottom();
    }

    getCurrentTime() {
        const now = new Date();
        let hours = now.getHours();
        let minutes = now.getMinutes();
        const ampm = hours >= 12 ? 'PM' : 'AM';
        
        hours = hours % 12;
        hours = hours ? hours : 12;
        minutes = minutes < 10 ? '0' + minutes : minutes;
        
        return `${hours}:${minutes} ${ampm}`;
    }

    showTypingIndicator() {
        this.typingIndicator.style.display = 'flex';
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }

    processMessage(message) {
        message = message.toLowerCase();
        let response = '';
        let foundMatch = false;

        // Check for keywords in message
        if (message.includes('admission') || message.includes('apply') || message.includes('join')) {
            response = this.knowledgeBase.admissions.response;
            foundMatch = true;
        }
        else if (message.includes('course') || message.includes('program') || message.includes('branch') || message.includes('degree')) {
            response = this.knowledgeBase.courses.response;
            foundMatch = true;
        }
        else if (message.includes('fee') || message.includes('cost') || message.includes('payment') || message.includes('money') || message.includes('expense')) {
            response = this.knowledgeBase.fees.response;
            foundMatch = true;
        }
        else if (message.includes('exam') || message.includes('test') || message.includes('schedule') || message.includes('result')) {
            response = this.knowledgeBase.exams.response;
            foundMatch = true;
        }
        else if (message.includes('placement') || message.includes('job') || message.includes('recruit') || message.includes('company') || message.includes('package')) {
            response = this.knowledgeBase.placement.response;
            foundMatch = true;
        }
        else if (message.includes('contact') || message.includes('phone') || message.includes('email') || message.includes('address') || message.includes('reach')) {
            response = this.knowledgeBase.contact.response;
            foundMatch = true;
        }
        else if (message.includes('facility') || message.includes('hostel') || message.includes('library') || message.includes('sports') || message.includes('lab')) {
            response = this.knowledgeBase.facilities.response;
            foundMatch = true;
        }
        else if (message.includes('hello') || message.includes('hi') || message.includes('hey')) {
            response = `Hello! 👋 Welcome to SITNAGPUR Official Chat Assistant.

I can help you with information about:
• Admissions 2026
• Courses & Programs
• Fee Structure & Scholarships
• Exam Schedule
• Placements & Internships
• Campus Facilities
• Contact Details

What would you like to know about?`;
            foundMatch = true;
        }
        else if (message.includes('thank')) {
            response = "You're welcome! 😊 Is there anything else I can help you with? Feel free to ask about any aspect of SITNAGPUR.";
            foundMatch = true;
        }
        else {
            response = `I'm not sure I understand your query. Could you please rephrase or select from these common topics:

• Admissions 2026
• Courses offered
• Fee structure
• Exam schedule
• Placements
• Campus facilities
• Contact information

Or you can type specific keywords like "B.Tech fees" or "placement statistics".`;
        }

        this.addMessage(response, 'bot');
    }

    handleQuickAction(query, event) {
        // Add user's quick action as message
        const buttonText = event.target.closest('.quick-action-btn').textContent.trim();
        this.addMessage(buttonText, 'user');

        // Show typing indicator
        this.showTypingIndicator();

        // Process and respond
        setTimeout(() => {
            this.hideTypingIndicator();
            
            let response = '';
            switch(query) {
                case 'admissions':
                    response = this.knowledgeBase.admissions.response;
                    break;
                case 'courses':
                    response = this.knowledgeBase.courses.response;
                    break;
                case 'fees':
                    response = this.knowledgeBase.fees.response;
                    break;
                case 'exams':
                    response = this.knowledgeBase.exams.response;
                    break;
                case 'placement':
                    response = this.knowledgeBase.placement.response;
                    break;
                case 'contact':
                    response = this.knowledgeBase.contact.response;
                    break;
                default:
                    response = "How can I help you with SITNAGPUR?";
            }
            
            this.addMessage(response, 'bot');
        }, 1500);
    }

    scrollToBottom() {
        this.messagesArea.scrollTop = this.messagesArea.scrollHeight;
    }
}

// Initialize chatbot when page loads
document.addEventListener('DOMContentLoaded', () => {
    new SITNAGPURChatbot();
});



// Enhanced version with some advanced features

class AdvancedSITNAGPURChatbot {
    constructor() {
        // Previous code...
        
        // Add context tracking
        this.conversationContext = {
            lastTopic: null,
            userDetails: {},
            followUpCount: 0,
            entities: {}
        };
        
        // Add entity patterns
        this.entityPatterns = {
            dates: [
                /\b(\d{1,2})(st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b/gi,
                /\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(st|nd|rd|th)?\b/gi,
                /\b(\d{1,2})[\/-](\d{1,2})[\/-](\d{2,4})\b/g
            ],
            courses: [
                /b\.?tech|btech|bachelor of technology/i,
                /m\.?tech|mtech|master of technology/i,
                /mba|master of business administration/i,
                /phd|doctor of philosophy/i,
                /computer science|cse|cs/i,
                /mechanical|me/i,
                /electronics|ece|ec/i,
                /civil|ce/i
            ]
        };
        
        // Analytics tracking
        this.analytics = {
            totalQueries: 0,
            successfulResponses: 0,
            fallbackCount: 0,
            popularTopics: {},
            userSatisfaction: []
        };
    }
    
    // Entity extraction method
    extractEntities(message) {
        const entities = {
            dates: [],
            courses: [],
            numbers: []
        };
        
        // Extract dates
        this.entityPatterns.dates.forEach(pattern => {
            const matches = message.matchAll(pattern);
            for (const match of matches) {
                entities.dates.push(match[0]);
            }
        });
        
        // Extract courses
        this.entityPatterns.courses.forEach(pattern => {
            if (pattern.test(message)) {
                const match = message.match(pattern);
                if (match) {
                    entities.courses.push(match[0]);
                }
            }
        });
        
        // Extract numbers (fees, percentages, etc.)
        const numberPattern = /\b(\d+(?:,\d+)?(?:\s?lakh|\s?crore)?)\b/g;
        const numberMatches = message.matchAll(numberPattern);
        for (const match of numberMatches) {
            entities.numbers.push(match[0]);
        }
        
        return entities;
    }
    
    // Context-aware response
    processMessageWithContext(message) {
        this.analytics.totalQueries++;
        
        // Extract entities
        const entities = this.extractEntities(message);
        
        // Check if it's a follow-up
        const isFollowUp = this.detectFollowUp(message);
        
        let response = '';
        
        // Use context if available
        if (isFollowUp && this.conversationContext.lastTopic) {
            response = this.handleFollowUp(message, this.conversationContext.lastTopic, entities);
        } else {
            // Normal message processing
            response = super.processMessage(message);
            
            // Update context
            this.updateContext(message, entities);
        }
        
        // Track analytics
        this.trackAnalytics(message, response);
        
        return response;
    }
    
    // Detect if message is a follow-up
    detectFollowUp(message) {
        const followUpIndicators = [
            'what about', 'tell me more', 'explain', 'elaborate',
            'and', 'also', 'how about', 'what else', 'more details',
            'can you', 'could you', 'please explain'
        ];
        
        return followUpIndicators.some(indicator => 
            message.toLowerCase().includes(indicator)
        ) || this.conversationContext.followUpCount > 0;
    }
    
    // Handle follow-up questions
    handleFollowUp(message, lastTopic, entities) {
        this.conversationContext.followUpCount++;
        
        // Different responses based on last topic
        switch(lastTopic) {
            case 'admissions':
                if (entities.dates.length > 0) {
                    return `Regarding admission dates: Applications are open from Jan 15 to Mar 31, 2026. You mentioned ${entities.dates.join(', ')} - would you like to know about specific deadlines?`;
                } else if (message.includes('fee') || message.includes('cost')) {
                    return `The application fee is ₹1200 for general category and ₹600 for SC/ST. Would you like to know about tuition fees as well?`;
                }
                break;
                
            case 'fees':
                if (entities.courses.length > 0) {
                    return `For ${entities.courses.join(', ')}, the fee structure varies. B.Tech is ₹1.25L/year, M.Tech is ₹1L/year. Which specific program are you interested in?`;
                } else if (message.includes('scholarship')) {
                    return `We offer merit-based scholarships up to 100% tuition waiver. Need-based scholarships up to 50% are also available. Would you like to know the eligibility criteria?`;
                }
                break;
                
            case 'placement':
                if (message.includes('company') || message.includes('recruiter')) {
                    return `Top recruiters include Microsoft, Google, Amazon, Goldman Sachs, Tata Motors, Infosys, and TCS. Would you like to see the placement statistics for a specific company?`;
                } else if (message.includes('package') || message.includes('salary')) {
                    return `The average package is ₹8.5 LPA, with the highest being ₹32 LPA. Top 10% of students get an average of ₹18 LPA.`;
                }
                break;
        }
        
        // Default follow-up response
        return `Following up on ${lastTopic}: ${this.knowledgeBase[lastTopic]?.response.substring(0, 100)}... Would you like more specific information?`;
    }
    
    // Update conversation context
    updateContext(message, entities) {
        // Detect main topic
        const topics = ['admissions', 'courses', 'fees', 'exams', 'placement', 'contact', 'facilities'];
        for (const topic of topics) {
            if (message.toLowerCase().includes(topic) || 
                (topic === 'admissions' && message.match(/apply|join|admission/i))) {
                this.conversationContext.lastTopic = topic;
                break;
            }
        }
        
        // Store entities
        if (entities.dates.length > 0) {
            this.conversationContext.entities.dates = entities.dates;
        }
        if (entities.courses.length > 0) {
            this.conversationContext.entities.courses = entities.courses;
        }
    }
    
    // Fallback to human handover
    checkForHandover(message) {
        const complexQueries = [
            'complaint', 'problem', 'issue', 'urgent',
            'talk to human', 'speak to agent', 'customer service',
            'not working', 'help me now', 'emergency'
        ];
        
        if (complexQueries.some(q => message.toLowerCase().includes(q))) {
            this.analytics.fallbackCount++;
            return {
                handover: true,
                message: "I understand this might require human assistance. Please hold while I connect you with our support team.\n\n📞 Support Team: +91 712 280 1234\n📧 Email: support@sitnagpur.edu.in\n⏱️ Estimated wait time: 2-3 minutes"
            };
        }
        
        return { handover: false };
    }
    
    // Analytics tracking
    trackAnalytics(query, response) {
        // Track topic popularity
        for (const topic in this.knowledgeBase) {
            if (query.toLowerCase().includes(topic)) {
                this.analytics.popularTopics[topic] = (this.analytics.popularTopics[topic] || 0) + 1;
            }
        }
        
        // Track success rate
        if (!response.includes("I'm not sure I understand")) {
            this.analytics.successfulResponses++;
        }
        
        // Log analytics (in production, send to backend)
        console.log('Analytics:', {
            totalQueries: this.analytics.totalQueries,
            successRate: (this.analytics.successfulResponses / this.analytics.totalQueries * 100).toFixed(2) + '%',
            popularTopics: this.analytics.popularTopics,
            fallbacks: this.analytics.fallbackCount
        });
    }
}