// ============================================================
// SITNAGPUR Chatbot — Complete Implementation
// All 10 Topics Covered
// ============================================================


// ============================================================
// TOPIC 2: Query Preprocessor
// Tokenization, Stop-word Removal, Stemming
// ============================================================
class QueryPreprocessor {
    constructor() {
        this.stopWords = new Set([
            'a','an','the','is','it','in','on','at','to','for','of','and','or',
            'but','not','with','from','by','this','that','are','was','were','be',
            'been','being','have','has','had','do','does','did','will','would',
            'could','should','may','might','shall','can','i','me','my','we','our',
            'you','your','he','she','they','their','what','which','who','when',
            'where','how','why','about','please','tell','give','want','need','know',
            'get','got','let','like','just','some','any','all','its','am','so','if'
        ]);

        this.stemmingRules = [
            { suffix: 'tion',  replacement: 'te'  },
            { suffix: 'ies',   replacement: 'y'   },
            { suffix: 'ied',   replacement: 'y'   },
            { suffix: 'ing',   replacement: ''    },
            { suffix: 'ness',  replacement: ''    },
            { suffix: 'ment',  replacement: ''    },
            { suffix: 'ed',    replacement: ''    },
            { suffix: 'er',    replacement: ''    },
            { suffix: 'est',   replacement: ''    },
            { suffix: 'ly',    replacement: ''    },
            { suffix: 'al',    replacement: ''    },
            { suffix: 's',     replacement: ''    },
        ];
    }

    tokenize(text) {
        return text.toLowerCase()
            .replace(/[^a-z0-9\s]/g, ' ')
            .split(/\s+/)
            .filter(t => t.length > 1);
    }

    removeStopWords(tokens) {
        return tokens.filter(t => !this.stopWords.has(t));
    }

    stem(word) {
        if (word.length <= 4) return word;
        for (const rule of this.stemmingRules) {
            if (word.endsWith(rule.suffix) && word.length - rule.suffix.length >= 3) {
                return word.slice(0, word.length - rule.suffix.length) + rule.replacement;
            }
        }
        return word;
    }

    // Full pipeline
    preprocess(text) {
        const tokens  = this.tokenize(text);
        const filtered = this.removeStopWords(tokens);
        const stemmed  = filtered.map(t => this.stem(t));
        return { original: text, tokens, filtered, stemmed };
    }
}


// ============================================================
// TOPIC 3: Synonym-Aware FAQ Bot
// Canonical synonym mapping
// ============================================================
class SynonymMapper {
    constructor() {
        this.synonymMap = {
            admission:  ['apply','application','joining','enroll','enrollment','entrance','register','registration','admit','intake','admission','admissions'],
            course:     ['program','programme','branch','degree','department','stream','subject','curriculum','major','course','courses'],
            fee:        ['cost','payment','expense','charge','tuition','money','price','amount','rupee','pay','afford','fees'],
            exam:       ['test','assessment','evaluation','examination','result','score','mark','grade','semester','paper','exam','exams'],
            placement:  ['job','recruit','recruitment','company','package','salary','career','hire','hiring','offer','intern','internship','placement','placements'],
            contact:    ['phone','call','email','address','reach','helpline','office','number','mail'],
            facility:   ['hostel','library','lab','laboratory','sports','cafeteria','canteen','infrastructure','building','campus','accommodation','gym','facility','facilities'],
            scholarship:['discount','waiver','concession','grant','financial','aid','stipend','merit','scholarship'],
        };

        this.reverseMap = {};
        for (const [canonical, synonyms] of Object.entries(this.synonymMap)) {
            for (const syn of synonyms) {
                this.reverseMap[syn] = canonical;
            }
        }
    }

    normalize(token) {
        return this.reverseMap[token] || token;
    }

    normalizeQuery(tokens) {
        return tokens.map(t => this.normalize(t));
    }

    findIntents(tokens) {
        const intents = new Set();
        for (const token of tokens) {
            const canonical = this.normalize(token);
            if (this.synonymMap[canonical]) intents.add(canonical);
        }
        return [...intents];
    }
}


// ============================================================
// TOPIC 4: TF-IDF FAQ Retrieval
// ============================================================
class TFIDFRetriever {
    constructor(faqs) {
        this.faqs = faqs;
        this.preprocessor = new QueryPreprocessor();
        this.idfScores = {};
        this._buildIDF();
    }

    _buildIDF() {
        const N = this.faqs.length;
        const df = {};
        for (const faq of this.faqs) {
            const { stemmed } = this.preprocessor.preprocess(faq.question + ' ' + faq.keywords);
            for (const term of new Set(stemmed)) {
                df[term] = (df[term] || 0) + 1;
            }
        }
        for (const [term, count] of Object.entries(df)) {
            this.idfScores[term] = Math.log((N + 1) / (count + 1)) + 1;
        }
    }

    _tf(terms) {
        const freq = {};
        for (const t of terms) freq[t] = (freq[t] || 0) + 1;
        const total = terms.length || 1;
        const tf = {};
        for (const t in freq) tf[t] = freq[t] / total;
        return tf;
    }

    _tfidfVec(terms) {
        const tf  = this._tf(terms);
        const vec = {};
        for (const [t, v] of Object.entries(tf)) {
            vec[t] = v * (this.idfScores[t] || 1);
        }
        return vec;
    }

    _cosine(v1, v2) {
        const all = new Set([...Object.keys(v1), ...Object.keys(v2)]);
        let dot = 0, m1 = 0, m2 = 0;
        for (const t of all) {
            const a = v1[t] || 0, b = v2[t] || 0;
            dot += a * b; m1 += a * a; m2 += b * b;
        }
        return m1 && m2 ? dot / (Math.sqrt(m1) * Math.sqrt(m2)) : 0;
    }

    retrieve(query, topK = 3) {
        const qVec = this._tfidfVec(this.preprocessor.preprocess(query).stemmed);
        return this.faqs
            .map(faq => {
                const fVec = this._tfidfVec(this.preprocessor.preprocess(faq.question + ' ' + faq.keywords).stemmed);
                return { faq, score: this._cosine(qVec, fVec) };
            })
            .sort((a, b) => b.score - a.score)
            .slice(0, topK)
            .filter(r => r.score > 0.05);
    }
}


// ============================================================
// TOPIC 5: Intent Classifier
// Multi-feature weighted scoring
// ============================================================
class IntentClassifier {
    constructor() {
        this.intents = {
            admissions: {
                keywords: ['admission','admissions','apply','application','enroll','joining','entrance','register','eligibility','deadline','document'],
                patterns:  [/when.*apply/i, /how.*apply/i, /apply.*college/i, /admission.*open/i, /eligib/i, /deadline/i, /last.*date/i],
                weight: 1.0
            },
            courses: {
                keywords: ['course','courses','program','branch','degree','btech','mtech','mba','phd','cse','ece','mechanical','civil','electrical'],
                patterns:  [/what.*course/i, /which.*program/i, /available.*branch/i, /b\.?tech/i, /m\.?tech/i, /department/i],
                weight: 1.0
            },
            fees: {
                keywords: ['fee','fees','cost','expense','payment','money','tuition','scholarship','concession','waiver'],
                patterns:  [/how.*much/i, /fee.*structure/i, /total.*cost/i, /scholarship/i, /afford/i, /rupee/i],
                weight: 1.0
            },
            exams: {
                keywords: ['exam','exams','test','schedule','result','mark','grade','semester','timetable','midterm','hall ticket','attendance'],
                patterns:  [/when.*exam/i, /exam.*date/i, /result.*out/i, /semester.*exam/i, /attendance/i, /revaluation/i],
                weight: 1.0
            },
            placement: {
                keywords: ['placement','placements','job','salary','package','company','recruit','career','hire','internship','offer','lpa'],
                patterns:  [/placement.*rate/i, /highest.*package/i, /company.*visit/i, /job.*offer/i, /average.*salary/i],
                weight: 1.0
            },
            contact: {
                keywords: ['contact','phone','email','address','reach','call','number','office','helpline'],
                patterns:  [/how.*contact/i, /phone.*number/i, /email.*address/i, /office.*hour/i, /where.*located/i],
                weight: 1.0
            },
            facilities: {
                keywords: ['facility','facilities','hostel','library','sports','lab','gym','cafeteria','campus','accommodation','wifi','infrastructure'],
                patterns:  [/hostel.*available/i, /library.*timing/i, /sports.*facility/i, /lab.*equipment/i, /campus.*life/i],
                weight: 1.0
            },
            greeting: {
                keywords: ['hello','hi','hey','namaste','morning','afternoon','evening','greet'],
                patterns:  [/^h(i|ey|ello)/i, /good\s+(morning|afternoon|evening)/i, /^namaste/i],
                weight: 0.8
            },
            thanks: {
                keywords: ['thank','thanks','thankyou','appreciate','helpful','great','awesome'],
                patterns:  [/thank/i, /appreciate/i, /that.*help/i],
                weight: 0.8
            }
        };
    }

    classify(preprocessed) {
        const { original, tokens } = preprocessed;
        const scores = {};

        for (const [intent, cfg] of Object.entries(this.intents)) {
            let score = 0;
            for (const kw of cfg.keywords) {
                if (original.toLowerCase().includes(kw)) score += 2;
                if (tokens.includes(kw)) score += 1;
            }
            for (const pat of cfg.patterns) {
                if (pat.test(original)) score += 3;
            }
            scores[intent] = score * cfg.weight;
        }

        const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
        const [topIntent, topScore] = sorted[0];
        const [, secondScore]       = sorted[1] || ['unknown', 0];

        return {
            intent:      topScore > 0 ? topIntent : 'unknown',
            score:       topScore,
            confidence:  topScore > 0 ? Math.min(topScore / 10, 1) : 0,
            allScores:   scores,
            isAmbiguous: topScore > 0 && secondScore > 0 && (topScore - secondScore) < 2
        };
    }
}


// ============================================================
// TOPIC 9: Multichannel Deployment Manager
// ============================================================
class MultichannelManager {
    constructor() {
        this.channels = {
            web:      { name: 'Web Chat',    icon: '🌐', active: true,  color: '#1e3c72' },
            whatsapp: { name: 'WhatsApp',    icon: '💬', active: false, color: '#25D366' },
            sms:      { name: 'SMS',         icon: '📱', active: false, color: '#FF6B35' },
            email:    { name: 'Email Bot',   icon: '📧', active: false, color: '#EA4335' }
        };
        this.activeChannel = 'web';
        this._renderChannelBar();
    }

    _renderChannelBar() {
        const header = document.querySelector('.chat-header');
        if (!header) return;

        const bar = document.createElement('div');
        bar.className = 'multichannel-bar';
        bar.innerHTML = `
            <div class="channel-label">Channels:</div>
            ${Object.entries(this.channels).map(([key, ch]) => `
                <button class="channel-btn ${ch.active ? 'active' : ''}"
                        data-channel="${key}" title="${ch.name}"
                        style="--ch-color:${ch.color}">
                    <span>${ch.icon}</span>
                    <span class="ch-name">${ch.name}</span>
                    <span class="ch-dot ${ch.active ? 'on' : 'off'}"></span>
                </button>
            `).join('')}
            <button class="deploy-trigger-btn" id="deployBtn">⚡ Deploy</button>
        `;
        header.appendChild(bar);

        bar.querySelectorAll('.channel-btn').forEach(btn =>
            btn.addEventListener('click', () => this._switchChannel(btn.dataset.channel))
        );
        document.getElementById('deployBtn').addEventListener('click', () => this._showModal());
    }

    _switchChannel(key) {
        this.activeChannel = key;
        document.querySelectorAll('.channel-btn').forEach(b =>
            b.classList.toggle('active', b.dataset.channel === key)
        );
        this._notify(this.channels[key]);
    }

    _notify(ch) {
        document.querySelector('.channel-notification')?.remove();
        const n = document.createElement('div');
        n.className = 'channel-notification';
        n.innerHTML = `${ch.icon} Switched to <strong>${ch.name}</strong> channel`;
        document.querySelector('.messages-area').prepend(n);
        setTimeout(() => n.remove(), 2500);
    }

    _showModal() {
        const overlay = document.createElement('div');
        overlay.className = 'deploy-overlay';
        overlay.innerHTML = `
            <div class="deploy-modal">
                <div class="deploy-modal-hdr">
                    <div>
                        <h2>⚡ Multichannel Deployment</h2>
                        <p>Deploy SITNAGPUR Assistant across all platforms</p>
                    </div>
                    <button class="modal-close" id="closeModal">✕</button>
                </div>

                <div class="deploy-grid">
                    ${Object.entries(this.channels).map(([key, ch]) => `
                        <div class="deploy-card">
                            <div class="deploy-card-icon" style="background:${ch.color}18;border:2px solid ${ch.color}40">${ch.icon}</div>
                            <h3>${ch.name}</h3>
                            <span class="deploy-status-badge ${ch.active ? 'live' : 'idle'}">${ch.active ? '✅ Live' : '⏳ Pending'}</span>
                            <button class="deploy-action-btn" data-ch="${key}"
                                    style="background:${ch.color}">
                                ${ch.active ? 'Manage' : 'Deploy Now'}
                            </button>
                        </div>
                    `).join('')}
                </div>

                <div class="embed-section">
                    <h3>📋 Web Widget Embed Code</h3>
                    <div class="code-snippet">
                        <code id="embedCode">&lt;script src="https://sitnagpur.edu.in/chatbot.js"&gt;&lt;/script&gt;<br>&lt;div id="sitnagpur-chat" data-theme="blue"&gt;&lt;/div&gt;</code>
                        <button class="copy-btn" id="copyEmbed">Copy</button>
                    </div>
                </div>

                <div class="deploy-footer-stats">
                    ${Object.entries(this.channels).map(([, ch]) =>
                        `<div class="df-stat">${ch.icon} ${ch.name}<br><small>${ch.active ? 'Active' : 'Ready to deploy'}</small></div>`
                    ).join('')}
                </div>
            </div>
        `;
        document.body.appendChild(overlay);

        document.getElementById('closeModal').onclick = () => overlay.remove();
        overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });

        overlay.querySelectorAll('.deploy-action-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const k = btn.dataset.ch;
                this.channels[k].active = !this.channels[k].active;
                const badge = btn.previousElementSibling;
                badge.textContent    = this.channels[k].active ? '✅ Live' : '⏳ Pending';
                badge.className      = `deploy-status-badge ${this.channels[k].active ? 'live' : 'idle'}`;
                btn.textContent      = this.channels[k].active ? 'Manage' : 'Deploy Now';
                // Sync channel bar dots
                document.querySelector(`.channel-btn[data-channel="${k}"] .ch-dot`)
                    ?.classList.toggle('on', this.channels[k].active);
            });
        });

        document.getElementById('copyEmbed').onclick = () => {
            navigator.clipboard.writeText(
                '<script src="https://sitnagpur.edu.in/chatbot.js"></script>\n<div id="sitnagpur-chat" data-theme="blue"></div>'
            );
            document.getElementById('copyEmbed').textContent = '✅ Copied!';
            setTimeout(() => { document.getElementById('copyEmbed').textContent = 'Copy'; }, 2000);
        };
    }
}


// ============================================================
// TOPIC 10: Live Analytics Dashboard
// ============================================================
class AnalyticsDashboard {
    constructor() {
        this.data = {
            totalQueries: 0, successfulResponses: 0,
            fallbackCount: 0, handoverCount: 0,
            sessionStart: new Date(), popularTopics: {},
            queryLog: [], unansweredQueries: [], responseTimings: []
        };
        this._render();
    }

    track(query, intent, success, responseTime) {
        this.data.totalQueries++;
        if (success) this.data.successfulResponses++;
        this.data.queryLog.push({ query, intent, success, time: new Date(), responseTime });
        if (intent && intent !== 'unknown') {
            this.data.popularTopics[intent] = (this.data.popularTopics[intent] || 0) + 1;
        }
        this.data.responseTimings.push(responseTime);
        if (!success) this.data.unansweredQueries.push(query);
        this._update();
    }

    trackFallback()  { this.data.fallbackCount++;  this._update(); }
    trackHandover()  { this.data.handoverCount++;   this._update(); }

    _successRate() {
        if (!this.data.totalQueries) return 0;
        return Math.round((this.data.successfulResponses / this.data.totalQueries) * 100);
    }

    _avgTime() {
        if (!this.data.responseTimings.length) return 0;
        return Math.round(this.data.responseTimings.reduce((a, b) => a + b, 0) / this.data.responseTimings.length);
    }

    _render() {
        const sidebar = document.querySelector('.institute-sidebar');
        if (!sidebar) return;
        const widget = document.createElement('div');
        widget.className = 'analytics-widget';
        widget.innerHTML = `
            <h3><i class="fas fa-chart-bar"></i> Live Analytics</h3>
            <div class="a-grid">
                <div class="a-stat"><span class="a-val" id="aTotal">0</span><span class="a-lbl">Queries</span></div>
                <div class="a-stat"><span class="a-val" id="aSuccess">0%</span><span class="a-lbl">Success</span></div>
                <div class="a-stat"><span class="a-val" id="aFallback">0</span><span class="a-lbl">Fallbacks</span></div>
                <div class="a-stat"><span class="a-val" id="aTime">0ms</span><span class="a-lbl">Avg Time</span></div>
            </div>
            <div class="a-topics" id="aTopics"><p class="a-empty">Ask something to see analytics…</p></div>
        `;
        sidebar.appendChild(widget);
    }

    _update() {
        const q = id => document.getElementById(id);
        if (q('aTotal'))   q('aTotal').textContent   = this.data.totalQueries;
        if (q('aSuccess')) q('aSuccess').textContent = this._successRate() + '%';
        if (q('aFallback')) q('aFallback').textContent = this.data.fallbackCount;
        if (q('aTime'))    q('aTime').textContent    = this._avgTime() + 'ms';

        const el = q('aTopics');
        if (!el) return;
        const sorted = Object.entries(this.data.popularTopics).sort((a, b) => b[1] - a[1]).slice(0, 5);
        if (!sorted.length) return;
        const max = sorted[0][1];
        el.innerHTML = `<p class="a-topics-lbl">Top Intents</p>` + sorted.map(([topic, count]) => `
            <div class="a-bar-row">
                <span class="a-bar-lbl">${topic}</span>
                <div class="a-bar-track"><div class="a-bar-fill" style="width:${(count/max)*100}%"></div></div>
                <span class="a-bar-count">${count}</span>
            </div>
        `).join('');
    }
}


// ============================================================
// TOPIC 1: Main Chatbot — integrates all modules
// ============================================================
class SITNAGPURChatbot {
    constructor() {
        // Topic 2, 3, 4, 5, 10
        this.preprocessor      = new QueryPreprocessor();
        this.synonymMapper     = new SynonymMapper();
        this.intentClassifier  = new IntentClassifier();
        this.analytics         = new AnalyticsDashboard();

        // Topic 7 — context
        this.context = {
            lastIntent: null, lastEntities: {},
            turnCount: 0, conversationHistory: []
        };

        // Topic 6 — entity patterns
        this.entityPatterns = {
            dates: [
                /\b(\d{1,2})(st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b/gi,
                /\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(st|nd|rd|th)?\b/gi,
                /\b(\d{1,2})[\/-](\d{1,2})[\/-](\d{2,4})\b/g
            ],
            courses: [
                /b\.?tech|btech|bachelor of technology/i,
                /m\.?tech|mtech|master of technology/i,
                /mba|master of business/i,
                /phd|doctor of philosophy/i,
                /computer science|cse|cs\b/i,
                /mechanical|mech\b/i,
                /electronics|ece\b/i,
                /civil\b/i,
                /ai\b|artificial intelligence|machine learning|ml\b/i,
                /data science/i,
                /robotics/i,
                /electrical\b/i
            ]
        };

        // Topic 1 — Knowledge Base (also seeds TF-IDF)
        this.knowledgeBase = {
            admissions: {
                title: "Admissions 2026",
                keywords: "apply join enroll eligibility deadline document certificate entrance gate cat mat registration open",
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
• Graduation certificates & entrance score card
• ID proof & passport size photographs

**Application Fee:** General: ₹1,200 | SC/ST: ₹600

Would you like to know about specific programs or scholarship opportunities?`
            },
            courses: {
                title: "Courses Offered",
                keywords: "btech mtech mba phd cse ece mechanical civil electrical ai ml data robotics program branch department specialization",
                response: `**Academic Programs at SITNAGPUR** 📚

**Undergraduate (B.Tech):**
• Computer Science & Engineering • Electronics & Communication
• Mechanical Engineering • Civil Engineering • Electrical Engineering
• Artificial Intelligence & ML • Data Science • Robotics & Automation

**Postgraduate (M.Tech):**
• AI & Machine Learning • VLSI Design • Structural Engineering
• Thermal Engineering • Computer Science

**Doctoral (PhD):**
• Engineering & Technology • Management Studies • Applied Sciences

**Management:**
• MBA in Technology Management • MBA in Business Analytics

Each program offers specialization options and industry-integrated curriculum.
Which program interests you?`
            },
            fees: {
                title: "Fee Structure",
                keywords: "tuition cost expense payment money rupee scholarship merit need based sc st concession waiver hostel mess library lab sports discount grant",
                response: `**Fee Structure for Academic Year 2026-27** 💰

**Tuition Fees (Per Year):**
• B.Tech: ₹1,25,000 | M.Tech: ₹1,00,000 | MBA: ₹1,50,000 | PhD: ₹75,000

**Additional Annual Fees:**
• Hostel (optional): ₹60,000–₹85,000 | Mess: ₹36,000
• Library: ₹5,000 | Laboratory: ₹8,000 | Sports: ₹2,000

**One-time Fees:**
• Admission: ₹10,000 | Caution Deposit (refundable): ₹5,000 | Alumni: ₹2,000

**Scholarship Opportunities:**
• Merit-based: Up to 100% tuition fee waiver
• Need-based: Up to 50% concession
• Sports quota: Up to 75% concession
• Girls scholarship: 10% additional discount
• SC/ST: As per government norms

Need help calculating total expenses or learning about payment plans?`
            },
            exams: {
                title: "Examination Schedule",
                keywords: "exam test schedule result score mark grade semester timetable midterm final practical supplementary revaluation attendance hall ticket",
                response: `**Examination Schedule 2026** 📝

**Mid-Term Exams:**
• Even Semester: March 10–20, 2026
• Odd Semester: September 15–25, 2026

**Final Exams:**
• Even Semester: May 5–25, 2026
• Odd Semester: November 20 – December 10, 2026

**Practical / Viva:**
• Even Semester: April 15–30, 2026
• Odd Semester: November 5–15, 2026

**Supplementary Exams:**
• Even Semester: July 10–20, 2026
• Odd Semester: January 10–20, 2027

**Key Guidelines:**
• Minimum 75% attendance required
• Hall tickets released 7 days before exams
• Results declared within 15 days
• Re-evaluation window: 7 days after results

Would you like to know more about exam rules or attendance?`
            },
            placement: {
                title: "Placement Statistics",
                keywords: "placement job recruit company package salary career hire offer internship microsoft google amazon goldman tcs infosys lpa average highest stipend",
                response: `**Placement Highlights 2025-26** 💼

**Overall Statistics:**
• Placement Rate: 92% | Average Package: ₹8.5 LPA
• Highest Package: ₹32 LPA | Top 10% Avg: ₹18 LPA | Total Offers: 450+

**Top Recruiters:**
• Microsoft: 12 offers (₹32 LPA) | Google: 8 offers (₹28 LPA)
• Amazon: 25 offers (₹24 LPA) | Goldman Sachs: 15 offers (₹20 LPA)
• Tata Motors: 30 offers (₹12 LPA) | L&T: 35 offers (₹9 LPA)
• Infosys: 45 offers (₹8 LPA) | TCS: 60 offers (₹7.5 LPA)

**Internships:**
• 85% students secured internships
• Average stipend: ₹25,000/month
• International internships: 25 students

**Upcoming Drives (March 2026):**
• Microsoft (7th) | Amazon (10th) | Google (15th) | Intel (20th)

Need help with placement preparation or company-wise details?`
            },
            contact: {
                title: "Contact Information",
                keywords: "phone call email address reach helpline office number mail emergency contact social media instagram linkedin facebook twitter",
                response: `**SITNAGPUR Contact Details** 📞

**Main Campus:**
📍 SIT Nagpur, Maharashtra – 440001
📞 +91 712 280 1234 | 📧 info@sitnagpur.edu.in

**Admissions Office:**
📞 +91 712 280 5678 | 📧 admissions@sitnagpur.edu.in | 🕒 Mon–Fri, 9 AM–5 PM

**Academic Office:**
📞 +91 712 280 9101 | 📧 academics@sitnagpur.edu.in

**Placement Cell:**
📞 +91 712 280 1122 | 📧 placement@sitnagpur.edu.in

**Emergency 24/7:**
🚑 +91 712 280 9999

**Social Media:**
Instagram: @sitnagpur_official | LinkedIn: SIT Nagpur | Twitter: @SITNagpur

Would you like directions to campus or to schedule a visit?`
            },
            facilities: {
                title: "Campus Facilities",
                keywords: "hostel library lab sports gym cafeteria canteen building campus accommodation wifi power backup smart classroom research innovation swimming pool basketball football",
                response: `**SITNAGPUR Campus Facilities** 🏛️

**Academic:**
• 25+ Advanced Laboratories | Central Library (50,000+ books)
• Smart Classrooms with IoT | Digital Learning Center
• Research & Innovation Hub

**Hostel:**
• Separate boys & girls hostels | Wi-Fi enabled campus
• 24/7 power backup | Gymnasium | Laundry services | Common rooms

**Sports Complex:**
• Olympic size swimming pool | Basketball & Tennis courts
• Football ground | Indoor badminton & Table tennis

**Amenities:**
• Medical center & ambulance | Cafeteria & Food court
• Banking & ATM | Transport services | 24/7 CCTV security

**Upcoming:** Innovation Incubation Center | E-Sports Arena | VR Lab

Want to know more about any specific facility?`
            }
        };

        // Topic 4 — TF-IDF index
        this.tfidfRetriever = new TFIDFRetriever(
            Object.entries(this.knowledgeBase).map(([id, kb]) => ({
                id, question: kb.title, keywords: kb.keywords, answer: kb.response
            }))
        );

        // DOM
        this.messagesArea   = document.getElementById('messagesArea');
        this.userInput      = document.getElementById('userInput');
        this.sendButton     = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');

        this._init();

        // Topic 9
        this.multichannel = new MultichannelManager();
    }

    _init() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.userInput.addEventListener('keypress', e => {
            if (e.key === 'Enter') this.sendMessage();
        });
        document.querySelectorAll('.quick-action-btn').forEach(btn =>
            btn.addEventListener('click', e => this._quickAction(e.currentTarget.dataset.query, e))
        );
        document.querySelectorAll('.chat-list-item').forEach(item =>
            item.addEventListener('click', function () {
                document.querySelectorAll('.chat-list-item').forEach(i => i.classList.remove('active'));
                this.classList.add('active');
            })
        );
        document.querySelector('.new-chat-btn')?.addEventListener('click', () => this._resetChat());
    }

    // ---- Topic 6: Entity Extraction ----
    _extractEntities(message) {
        const entities = { dates: [], courses: [], numbers: [] };

        this.entityPatterns.dates.forEach(pat => {
            const matches = [...message.matchAll(pat)];
            matches.forEach(m => entities.dates.push(m[0]));
        });
        this.entityPatterns.courses.forEach(pat => {
            const m = message.match(pat);
            if (m) entities.courses.push(m[0]);
        });
        const nums = [...message.matchAll(/\b(\d+(?:,\d+)?(?:\s?lakh|\s?crore)?)\b/g)];
        nums.forEach(m => entities.numbers.push(m[0]));

        return entities;
    }

    // ---- Topic 7: Context & Follow-up Handling ----
    _isFollowUp(message, preprocessed) {
        const phrases = ['what about','tell me more','explain','and what','also','how about','what else','more details','yes please','go on','elaborate','can you tell'];
        const hasPhrase  = phrases.some(p => message.toLowerCase().includes(p));
        const isShort    = preprocessed.filtered.length <= 2;
        return (hasPhrase || isShort) && this.context.lastIntent !== null;
    }

    _handleFollowUp(message, entities) {
        const last = this.context.lastIntent;
        this.context.turnCount++;

        const msg = message.toLowerCase();

        if (last === 'admissions') {
            if (entities.dates.length)        return `Our application window is Jan 15 – Mar 31, 2026. You mentioned **${entities.dates.join(', ')}** — want specific deadlines for that period?`;
            if (msg.match(/fee|cost|money/))  return `The **application fee** is ₹1,200 (General) and ₹600 (SC/ST). Want details on tuition fees and scholarships too?`;
            if (msg.match(/document|certificate|required/)) return `You'll need 10th & 12th marksheets, graduation certificates, entrance score card, ID proof, and passport photos. Shall I elaborate?`;
        }
        if (last === 'fees') {
            if (entities.courses.length)      return `For **${entities.courses.join(', ')}**: B.Tech ₹1.25L/yr, M.Tech ₹1L/yr, MBA ₹1.5L/yr. Which program specifically?`;
            if (msg.match(/scholarship|waiver|discount/)) return `Scholarships: Merit (up to 100%), Need-based (50%), Sports (75%), Girls (10% extra), SC/ST govt norms. Want eligibility criteria?`;
        }
        if (last === 'placement') {
            if (msg.match(/company|recruiter/)) return `Top recruiters: **Microsoft** (₹32 LPA), **Google** (₹28 LPA), **Amazon** (₹24 LPA), **Goldman Sachs** (₹20 LPA). Which company do you want stats for?`;
            if (msg.match(/package|salary|ctc/)) return `Average: **₹8.5 LPA**, Highest: ₹32 LPA, Top 10% avg: ₹18 LPA. Want branch-specific placement data?`;
        }
        if (last === 'exams') {
            if (msg.match(/result|mark|grade/)) return `Results are declared within **15 days** of exams. Re-evaluation requests can be submitted within 7 days of results. Need specific dates?`;
            if (msg.match(/attendance/))        return `Minimum **75% attendance** is required to be eligible for exams. Shortage of attendance cases are handled by the Academic Office.`;
        }
        if (last === 'facilities') {
            if (msg.match(/hostel|accommodation/)) return `Both boys' and girls' hostels are available with Wi-Fi, 24/7 power backup, gym, and laundry. Annual cost: ₹60,000–₹85,000.`;
            if (msg.match(/library/))              return `The central library has 50,000+ books, digital resources, e-journals, and is open 8 AM – 10 PM on weekdays.`;
        }

        const kb = this.knowledgeBase[last];
        return kb ? `Here's more on **${kb.title}**:\n\n${kb.response}` : null;
    }

    _updateContext(intent, entities) {
        this.context.lastIntent = intent;
        this.context.lastEntities = entities;
        this.context.turnCount++;
        this.context.conversationHistory.push({ intent, entities, time: new Date() });
    }

    // ---- Topic 8: Fallback & Handover ----
    _checkHandover(message) {
        const triggers = ['complaint','problem','issue','urgent','talk to human','speak to agent','real person','customer service','not working','help me now','emergency','escalate'];
        if (triggers.some(t => message.toLowerCase().includes(t))) {
            this.analytics.trackHandover();
            return {
                triggered: true,
                response: `I understand this needs immediate attention. Connecting you with our support team. 🔄

📞 **Support Hotline:** +91 712 280 1234
📧 **Email:** support@sitnagpur.edu.in
🚑 **Emergency 24/7:** +91 712 280 9999
⏱️ **Estimated wait:** 2–3 minutes

Available Mon–Sat, 9 AM–5 PM. For emergencies, the 24/7 line is always open.`
            };
        }
        return { triggered: false };
    }

    _fallbackResponse(tfidfResults) {
        this.analytics.trackFallback();
        if (tfidfResults && tfidfResults.length > 0) {
            const suggestions = tfidfResults.map(r =>
                `• **${r.faq.id.charAt(0).toUpperCase() + r.faq.id.slice(1)}** — ${this.knowledgeBase[r.faq.id]?.title || ''}`
            ).join('\n');
            return `I'm not entirely sure about that, but you might be looking for:\n\n${suggestions}\n\nOr try rephrasing with keywords like "B.Tech fees" or "placement statistics".`;
        }
        return `I'm not sure I understand. You can ask about:\n\n• **Admissions 2026** — eligibility, dates, documents\n• **Courses** — B.Tech, M.Tech, MBA, PhD\n• **Fee Structure** — tuition, scholarships\n• **Exam Schedule** — mid-term, finals, results\n• **Placements** — statistics, companies, packages\n• **Facilities** — hostel, library, sports\n• **Contact** — phone, email, address`;
    }

    // ---- Core Processing Pipeline ----
    processMessage(message) {
        const t0 = Date.now();

        // Step 1 — Handover check (Topic 8)
        const handover = this._checkHandover(message);
        if (handover.triggered) {
            this.addMessage(handover.response, 'bot');
            return;
        }

        // Step 2 — Preprocess (Topic 2)
        const preprocessed = this.preprocessor.preprocess(message);

        // Step 3 — Synonym normalization (Topic 3)
        const normalizedTokens = this.synonymMapper.normalizeQuery(preprocessed.stemmed);

        // Step 4 — Entity extraction (Topic 6)
        const entities = this._extractEntities(message);

        // Step 5 — Follow-up / context (Topic 7)
        if (this._isFollowUp(message, preprocessed)) {
            const followUpResp = this._handleFollowUp(message, entities);
            if (followUpResp) {
                this.analytics.track(message, this.context.lastIntent, true, Date.now() - t0);
                this.addMessage(followUpResp, 'bot');
                return;
            }
        }

        // Step 6 — Intent classification (Topic 5)
        const classification = this.intentClassifier.classify(preprocessed);

        // Step 7 — TF-IDF retrieval as backup (Topic 4)
        const tfidfResults = this.tfidfRetriever.retrieve(message);

        let response = '';
        let success  = true;
        let intent   = classification.intent;

        if (intent === 'greeting') {
            response = `Hello! 👋 Welcome to SITNAGPUR Official Chat Assistant.\n\nI can help you with:\n• Admissions 2026 | Courses & Programs | Fee Structure & Scholarships\n• Exam Schedule | Placements & Internships | Campus Facilities | Contact\n\nWhat would you like to know about?`;
        } else if (intent === 'thanks') {
            response = "You're welcome! 😊 Feel free to ask if you need anything else about SITNAGPUR.";
        } else if (this.knowledgeBase[intent]) {
            response = this.knowledgeBase[intent].response;
            this._updateContext(intent, entities);
        } else if (tfidfResults.length > 0) {
            // TF-IDF retrieval fallback (Topic 4)
            const best = tfidfResults[0];
            if (best.score > 0.12 && this.knowledgeBase[best.faq.id]) {
                intent   = best.faq.id;
                response = this.knowledgeBase[best.faq.id].response;
                this._updateContext(intent, entities);
            } else {
                response = this._fallbackResponse(tfidfResults);
                success  = false;
            }
        } else {
            response = this._fallbackResponse(null);
            success  = false;
        }

        this.analytics.track(message, intent, success, Date.now() - t0);
        this.addMessage(response, 'bot');
    }

    sendMessage() {
        const msg = this.userInput.value.trim();
        if (!msg) return;
        this.addMessage(msg, 'user');
        this.userInput.value = '';
        this._showTyping();
        setTimeout(() => {
            this._hideTyping();
            this.processMessage(msg);
        }, 900 + Math.random() * 400);
    }

    _quickAction(query, event) {
        const label = event.target.closest('.quick-action-btn').textContent.trim();
        this.addMessage(label, 'user');
        this._showTyping();
        setTimeout(() => {
            this._hideTyping();
            if (this.knowledgeBase[query]) {
                this._updateContext(query, {});
                this.analytics.track(label, query, true, 80);
                this.addMessage(this.knowledgeBase[query].response, 'bot');
            }
        }, 900);
    }

    addMessage(text, sender) {
        const wrap = document.createElement('div');
        wrap.className = `message ${sender}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = sender === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
        wrap.appendChild(avatar);

        const contentWrap = document.createElement('div');
        contentWrap.className = 'message-content-wrapper';

        const senderEl = document.createElement('div');
        senderEl.className = 'message-sender';
        senderEl.textContent = sender === 'bot' ? 'SITNAGPUR Assistant' : 'You';
        contentWrap.appendChild(senderEl);

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
        contentWrap.appendChild(bubble);

        const time = document.createElement('span');
        time.className = 'message-time';
        time.textContent = this._time();
        contentWrap.appendChild(time);

        wrap.appendChild(contentWrap);
        this.messagesArea.appendChild(wrap);
        wrap.style.animation = 'fadeInUp 0.3s ease forwards';
        this._scrollBottom();
    }

    _resetChat() {
        this.messagesArea.querySelectorAll('.message, .quick-actions-container')
            .forEach(el => el.remove());
        this.context = { lastIntent: null, lastEntities: {}, turnCount: 0, conversationHistory: [] };
    }

    _time() {
        const d = new Date();
        let h = d.getHours(), m = d.getMinutes();
        const ap = h >= 12 ? 'PM' : 'AM';
        h = h % 12 || 12;
        return `${h}:${m < 10 ? '0' + m : m} ${ap}`;
    }

    _showTyping()  { this.typingIndicator.style.display = 'flex'; this._scrollBottom(); }
    _hideTyping()  { this.typingIndicator.style.display = 'none'; }
    _scrollBottom(){ this.messagesArea.scrollTop = this.messagesArea.scrollHeight; }
}

// Boot
document.addEventListener('DOMContentLoaded', () => new SITNAGPURChatbot());
