
// API Configuration
const API_BASE = 'http://localhost:8000/api';

// ═══════════════════ DATA ENGINE ═══════════════════
const R = (min, max) => Math.random() * (max - min) + min;
const fmt = (n,d=2) => n.toFixed(d);
const fmtN = n => n >= 1e7 ? (n/1e7).toFixed(2)+'Cr' : n >= 1e5 ? (n/1e5).toFixed(1)+'L' : n.toFixed(0);
const fmtP = n => (n>=0?'+':'')+fmt(n)+'%';

// Global data
let stockPrices = {};
let currentStock = null;
let stockChart = null;
let currentSector = 'all';
let currentSearch = '';
let predFilter = 'all';

// ═══════════════════ API CALLS ═══════════════════
async function fetchAPI(endpoint) {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`);
    if (!response.ok) {
      if (response.status === 404) return null; // Gracefully handle 404 without console noise
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`API Error (${endpoint}):`, error);
    return null;
  }
}

// ═══════════════════ TICKER BAR ═══════════════════
async function buildTicker() {
  const data = await fetchAPI('/ticker');
  if (!data) return;
  
  const bar = document.getElementById('tickerBar');
  bar.innerHTML = data.map(t=>`
    <div class="tick-item">
      <span class="tick-name">${t.name}</span>
      <span class="tick-val ${t.up?'up':'dn'}">${t.value}</span>
      <span class="tick-chg ${t.up?'up-bg':'dn-bg'}">${t.change}</span>
    </div>
  `).join('');
}

// Clock + market status
function updateClock() {
  const now = new Date();
  const ist = new Date(now.toLocaleString('en',{timeZone:'Asia/Kolkata'}));
  const h = ist.getHours(), m = ist.getMinutes();
  document.getElementById('clock').textContent =
    `IST ${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}`;
  const isOpen = h >= 9 && (h < 15 || (h===15&&m<30));
  const ms = document.getElementById('marketStatus');
  ms.textContent = isOpen ? '● LIVE' : '○ CLOSED';
  ms.className = 'market-status ' + (isOpen ? 'status-open':'status-closed');
}

// ═══════════════════ HOME VIEW ═══════════════════
async function buildHome() {
  // Indices
  const indices = await fetchAPI('/indices');
  if (indices) {
    const ir = document.getElementById('indicesRow');
    ir.innerHTML = indices.map(idx => {
      const up = idx.change >= 0;
      return `<div class="idx-card">
        <div class="idx-name">${idx.name}</div>
        <div class="idx-val">${idx.value.toLocaleString('en-IN',{maximumFractionDigits:2})}</div>
        <div class="idx-row">
          <span class="idx-chg ${up?'up':'dn'}">${up?'+':''}${fmt(idx.change)}%</span>
          <canvas class="idx-mini" id="mini_${idx.id}"></canvas>
        </div>
      </div>`;
    }).join('');
    
    // Draw mini sparklines
    setTimeout(() => {
      indices.forEach(idx => {
        const c = document.getElementById('mini_'+idx.id);
        if(!c) return;
        const ctx = c.getContext('2d');
        const pts = Array.from({length:20}, (_,i) => idx.value);
        const mn = Math.min(...pts), mx = Math.max(...pts);
        const w = c.offsetWidth||80, h = c.offsetHeight||28;
        c.width = w; c.height = h;
        ctx.strokeStyle = idx.change>=0 ? '#10b981' : '#ef4444';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        pts.forEach((v,i) => {
          const x = (i/(pts.length-1))*w;
          const y = h - ((v-mn)/(mx-mn||1))*h;
          i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
        });
        ctx.stroke();
      });
    }, 50);
  }

  // Sector heatmap
  const sectors = await fetchAPI('/sectors');
  if (sectors) {
    document.getElementById('sectorHeat').innerHTML = sectors.map(s=>{
      const intensity = Math.min(Math.abs(s.change)/3,1);
      const bg = s.change>=0
        ? `rgba(16,185,129,${0.2+intensity*0.6})` 
        : `rgba(239,68,68,${0.2+intensity*0.6})`;
      return `<div class="heat-cell" style="background:${bg}" onclick="filterSectorFromHome('${s.name}')">
        <div class="heat-name">${s.name}</div>
        <div class="heat-chg">${s.change>=0?'+':''}${fmt(s.change)}%</div>
      </div>`;
    }).join('');
  }

  // Gainers/Losers
  const stocks = await fetchAPI('/stocks');
  if (stocks) {
    stocks.forEach(s => {
      stockPrices[s.symbol] = {
        price: s.price,
        chg: s.change_pct,
        vol: s.volume,
        signal: s.signal,
        high52: s.high_52w,
        low52: s.low_52w,
      };
    });

    const gainers = [...stocks].sort((a,b)=>b.change_pct-a.change_pct).slice(0,7);
    const losers  = [...stocks].sort((a,b)=>a.change_pct-b.change_pct).slice(0,7);

    document.getElementById('gainersTable').innerHTML =
      '<thead><tr><th>Symbol</th><th>Price</th><th>Chg%</th></tr></thead>' +
      '<tbody>' + gainers.map(s=>`
        <tr class="clickable" onclick="openStockFromHome('${s.symbol}')">
          <td><div class="sym-cell"><div class="sym-dot" style="background:var(--green)"></div>${s.symbol}</div></td>
          <td>₹${fmt(s.price)}</td>
          <td class="up">+${fmt(s.change_pct)}%</td>
        </tr>`).join('') + '</tbody>';

    document.getElementById('losersTable').innerHTML =
      '<thead><tr><th>Symbol</th><th>Price</th><th>Chg%</th></tr></thead>' +
      '<tbody>' + losers.map(s=>`
        <tr class="clickable" onclick="openStockFromHome('${s.symbol}')">
          <td><div class="sym-cell"><div class="sym-dot" style="background:var(--red)"></div>${s.symbol}</div></td>
          <td>₹${fmt(s.price)}</td>
          <td class="dn">${fmt(s.change_pct)}%</td>
        </tr>`).join('') + '</tbody>';
  }

  // Market breadth
  const all = Object.values(stockPrices);
  const adv = all.filter(s=>s.chg>0).length;
  const dec = all.filter(s=>s.chg<0).length;
  const unch = all.length - adv - dec;
  const bc = document.getElementById('breadthChart');
  if(bc && !bc._built) {
    bc._built = true;
    new Chart(bc.getContext('2d'), {
      type:'doughnut',
      data:{
        labels:['Advancing','Declining','Unchanged'],
        datasets:[{data:[adv,dec,unch],backgroundColor:['#10b981','#ef4444','#6b7280'],borderWidth:0,hoverOffset:4}]
      },
      options:{plugins:{legend:{position:'right',labels:{color:'#9ca3af',font:{size:10},boxWidth:10}}},cutout:'65%',responsive:true,maintainAspectRatio:false}
    });
  }

  // F&O Activity
  document.getElementById('fnoActivity').innerHTML = [
    {l:'NIFTY PCR', v:'0.87', n:true},
    {l:'BANKNIFTY PCR', v:'0.94', n:true},
    {l:'Max Pain NIFTY', v:'22,000', n:false},
    {l:'Call OI (Cr)', v:'3,42,847', n:false},
    {l:'Put OI (Cr)', v:'2,98,432', n:false},
    {l:'IV Rank NIFTY', v:'34%', n:false},
  ].map(m=>`
    <div class="metric-row">
      <span class="metric-key">${m.l}</span>
      <span class="metric-val" style="color:var(--cyan)">${m.v}</span>
    </div>`).join('');

  // 52W Stats
  document.getElementById('weekStats').innerHTML = [
    {l:'At 52W High', v:'47', c:'var(--green)'},
    {l:'At 52W Low', v:'12', c:'var(--red)'},
    {l:'Near High (5%)', v:'143', c:'var(--amber)'},
    {l:'Above 200 DMA', v:'68%', c:'var(--green)'},
    {l:'Above 50 DMA', v:'54%', c:'var(--amber)'},
  ].map(m=>`
    <div class="metric-row">
      <span class="metric-key">${m.l}</span>
      <span class="metric-val" style="color:${m.c}">${m.v}</span>
    </div>`).join('');
}

// ═══════════════════ MARKETS VIEW ═══════════════════
async function buildStocksTable(filter='all', search='') {
  const stocks = await fetchAPI('/stocks');
  if (!stocks) return;
  
  let filteredStocks = stocks;
  if(filter !== 'all') filteredStocks = stocks.filter(s=>s.sector===filter);
  if(search) filteredStocks = stocks.filter(s=>
    s.symbol.toLowerCase().includes(search.toLowerCase()) ||
    s.name.toLowerCase().includes(search.toLowerCase())
  );
  
  document.getElementById('stockCount').textContent = `${filteredStocks.length} stocks`;
  
  const tbody = document.getElementById('stockTableBody');
  tbody.innerHTML = filteredStocks.map(s => {
    const up = s.change_pct >= 0;
    const sig = s.signal;
    const sigC = sig==='BUY'?'sig-buy':sig==='SELL'?'sig-sell':'sig-neutral';
    const pct52 = ((s.price - s.low_52w)/(s.high_52w - s.low_52w)*100);
    return `<tr class="clickable" onclick="openStockDetail('${s.symbol}')">
      <td><div class="sym-cell"><div class="sym-dot" style="background:${up?'var(--green)':'var(--red)'}"></div><b>${s.symbol}</b></div></td>
      <td style="color:var(--muted2);font-size:11px">${s.name.substring(0,20)}</td>
      <td style="color:${up?'var(--green)':'var(--red)'}">₹${fmt(s.price)}</td>
      <td class="${up?'up':'dn'}">${up?'+':''}${fmt(s.change,2)}</td>
      <td class="${up?'up':'dn'}">${up?'+':''}${fmt(s.change_pct)}%</td>
      <td>${fmtN(s.volume)}</td>
      <td style="color:var(--muted2)">${s.market_cap}</td>
      <td><span class="ind-sig ${sigC}">${sig}</span></td>
      <td>
        <div style="width:60px;height:4px;background:var(--bg4);border-radius:2px;overflow:hidden">
          <div style="width:${pct52}%;height:100%;background:${pct52>70?'var(--green)':pct52<30?'var(--red)':'var(--amber)'}"></div>
        </div>
      </td>
    </tr>`;
  }).join('');
}

function filterStocks(v) { currentSearch=v; buildStocksTable(currentSector,v); }
function filterSector(s,btn) {
  currentSector=s;
  document.querySelectorAll('.tabs .tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  buildStocksTable(s,currentSearch);
}
function filterSectorFromHome(s) {
  showView('markets');
  currentSector=s;
  buildStocksTable(s,'');
}
function openStockFromHome(sym) { showView('markets'); openStockDetail(sym); }

// ═══════════════════ STOCK DETAIL ═══════════════════
async function openStockDetail(sym) {
  currentStock = sym;
  const stock = await fetchAPI(`/stocks/${sym}`);
  if (!stock) return;
  
  document.getElementById('stockList').style.display='none';
  const det = document.getElementById('stock-detail');
  det.style.display='flex'; det.classList.add('active');

  document.getElementById('detailSymbol').textContent = stock.symbol;
  document.getElementById('detailName').textContent = stock.name;
  document.getElementById('detailPrice').textContent = '₹'+fmt(stock.price);
  const up = stock.change_pct>=0;
  document.getElementById('detailChange').innerHTML =
    `<span class="${up?'up':'dn'}">${up?'+':''}${fmt(stock.change_pct)}% (${up?'+':''}₹${fmt(stock.change)})</span>`;
  document.getElementById('detailOpen').textContent = '₹'+fmt(stock.price*R(0.995,1.005));
  document.getElementById('detailHigh').textContent = '₹'+fmt(stock.price*R(1.005,1.025));
  document.getElementById('detailLow').textContent  = '₹'+fmt(stock.price*R(0.975,0.995));
  document.getElementById('detailVol').textContent  = fmtN(stock.volume);

  buildStockChart(sym);
  buildIndicators(sym);
  buildPatterns();
  buildPredictionCards(sym);
  buildStockNews(sym);
  buildIndicatorStrip(sym);
}

function closeStockDetail() {
  document.getElementById('stockList').style.display='flex';
  const det = document.getElementById('stock-detail');
  det.style.display='none'; det.classList.remove('active');
  if(stockChart){stockChart.destroy();stockChart=null;}
}

function buildStockChart(sym) {
  const stock = stockPrices[sym];
  if (!stock) return;
  
  const n = 60;
  const labels = Array.from({length:n},(_,i)=>{
    const t = new Date(); t.setMinutes(t.getMinutes()-n+i);
    return `${String(t.getHours()).padStart(2,'0')}:${String(t.getMinutes()).padStart(2,'0')}`;
  });
  let prices = [stock.price];
  for(let i=1;i<n;i++) prices.push(prices[i-1]*(1+R(-0.003,0.003)));
  prices[prices.length-1] = stock.price;

  // EMA 20
  const ema20 = []; let k=2/21;
  prices.forEach((p,i) => ema20.push(i===0?p:(p*k+ema20[i-1]*(1-k))));

  // Bollinger Bands
  const bbUpper=[], bbLower=[], bbMid=[];
  prices.forEach((p,i) => {
    const w = prices.slice(Math.max(0,i-19),i+1);
    const m = w.reduce((a,b)=>a+b,0)/w.length;
    const std = Math.sqrt(w.reduce((a,b)=>a+(b-m)**2,0)/w.length);
    bbMid.push(m); bbUpper.push(m+2*std); bbLower.push(m-2*std);
  });

  const canvas = document.getElementById('stockChart');
  if(stockChart){stockChart.destroy();}
  stockChart = new Chart(canvas.getContext('2d'), {
    type:'line',
    data:{
      labels,
      datasets:[
        {label:'Price',data:prices,borderColor:'#3b82f6',borderWidth:2,pointRadius:0,tension:0.2,fill:false,order:1},
        {label:'EMA20',data:ema20,borderColor:'#f59e0b',borderWidth:1,borderDash:[],pointRadius:0,tension:0.2,fill:false,order:2},
        {label:'BB Upper',data:bbUpper,borderColor:'rgba(139,92,246,.4)',borderWidth:1,pointRadius:0,tension:0.2,fill:false,order:3,borderDash:[3,3]},
        {label:'BB Lower',data:bbLower,borderColor:'rgba(139,92,246,.4)',borderWidth:1,pointRadius:0,tension:0.2,fill:'+1',backgroundColor:'rgba(139,92,246,.05)',order:4,borderDash:[3,3]},
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      interaction:{mode:'index',intersect:false},
      plugins:{legend:{display:true,labels:{color:'#6b7280',font:{size:9},boxWidth:12}}},
      scales:{
        x:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#6b7280',font:{size:9},maxTicksLimit:12}},
        y:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#6b7280',font:{size:9},callback:v=>'₹'+v.toFixed(0)},position:'right'}
      }
    }
  });
}

function setTF(btn, tf) {
  document.querySelectorAll('.tf-btn').forEach(b=>{ if(['1D','1W','1M','3M','1Y'].includes(b.textContent)) b.classList.remove('active'); });
  btn.classList.add('active');
  if(currentStock) buildStockChart(currentStock);
}

function buildIndicatorStrip(sym) {
  const d = stockPrices[sym];
  if (!d) return;
  
  const p = d.price;
  const items = [
    {n:'RSI(14)', v:fmt(R(30,75)), sig: (v=>v>70?'OVERBOUGHT':v<30?'OVERSOLD':'NEUTRAL')(R(30,75))},
    {n:'MACD', v:fmt(R(-5,5)), sig:R(0,1)>0.5?'BULLISH':'BEARISH'},
    {n:'MFI(14)', v:fmt(R(30,80)), sig:'NEUTRAL'},
    {n:'ADX(14)', v:fmt(R(15,45)), sig:R(0,1)>0.6?'STRONG':'WEAK'},
    {n:'ATR(14)', v:'₹'+fmt(p*R(0.005,0.025)), sig:'–'},
  ];
  document.getElementById('indicatorStrip').innerHTML = items.map(i=>`
    <div style="background:var(--bg4);border-radius:5px;padding:6px 8px;text-align:center">
      <div style="font-size:9px;color:var(--muted);margin-bottom:2px">${i.n}</div>
      <div style="font-family:var(--mono);font-size:12px;font-weight:700">${i.v}</div>
      <div style="font-size:9px;color:var(--muted2);margin-top:1px">${i.sig}</div>
    </div>`).join('');
}

function buildIndicators(sym) {
  const p = stockPrices[sym].price;
  const inds = [
    {n:'RSI (14)',   v:fmt(R(35,72)),  sig:()=>{const v=R(35,72);return v>65?'SELL':v<40?'BUY':'NEUTRAL'}},
    {n:'MACD',       v:fmt(R(-8,8)),   sig:()=>R(0,1)>0.55?'BUY':'SELL'},
    {n:'EMA 20',     v:'₹'+fmt(p*R(0.98,1.02)), sig:()=>R(0,1)>0.5?'BUY':'NEUTRAL'},
    {n:'EMA 50',     v:'₹'+fmt(p*R(0.95,1.05)), sig:()=>R(0,1)>0.45?'BUY':'SELL'},
    {n:'EMA 200',    v:'₹'+fmt(p*R(0.85,1.15)), sig:()=>'NEUTRAL'},
    {n:'Bollinger',  v:'Mid ₹'+fmt(p), sig:()=>R(0,1)>0.6?'BUY':'NEUTRAL'},
    {n:'Stoch RSI',  v:fmt(R(20,85))+'%',sig:()=>{const v=R(20,85);return v>80?'SELL':v<20?'BUY':'NEUTRAL'}},
    {n:'ADX (14)',   v:fmt(R(18,42)),  sig:()=>R(0,1)>0.6?'BUY':'NEUTRAL'},
    {n:'CCI (20)',   v:fmt(R(-150,150)),sig:()=>{const v=R(-150,150);return v>100?'SELL':v<-100?'BUY':'NEUTRAL'}},
    {n:'Williams %R',v:'-'+fmt(R(10,90))+'%',sig:()=>{const v=R(10,90);return v<20?'SELL':v>80?'BUY':'NEUTRAL'}},
    {n:'OBV',        v:fmtN(R(1e6,1e8)),sig:()=>R(0,1)>0.5?'BUY':'NEUTRAL'},
    {n:'VWAP',       v:'₹'+fmt(p*R(0.99,1.01)),sig:()=>R(0,1)>0.55?'BUY':'SELL'},
  ];
  document.getElementById('techIndicators').innerHTML = inds.map(i=>{
    const s = i.sig();
    const sc = s==='BUY'?'sig-buy':s==='SELL'?'sig-sell':'sig-neutral';
    return `<div class="ind-row">
      <span class="ind-name">${i.n}</span>
      <span class="ind-val">${i.v}</span>
      <span class="ind-sig ${sc}">${s}</span>
    </div>`;
  }).join('');
}

function buildPatterns() {
  const chartPats = [
    {n:'Bull Flag',      c:'var(--green)',  s:'Forming',  bull:true,  str:'Strong'},
    {n:'Cup & Handle',   c:'var(--green)',  s:'Detected', bull:true,  str:'Medium'},
    {n:'Head & Shoulder',c:'var(--amber)',  s:'Possible', bull:false, str:'Weak'},
    {n:'Ascending Triangle',c:'var(--green)',s:'Forming', bull:true,  str:'Strong'},
    {n:'Double Bottom',  c:'var(--cyan)',   s:'Confirmed',bull:true,  str:'Strong'},
  ];
  document.getElementById('chartPatterns').innerHTML = chartPats.map(p=>`
    <div class="pat-item">
      <div class="pat-icon" style="background:${p.bull?'var(--green2)':'var(--red2)'};color:${p.c}">${p.bull?'▲':'▼'}</div>
      <span class="pat-name">${p.n}</span>
      <span class="pat-strength" style="color:${p.c}">${p.s}</span>
    </div>`).join('');

  const candlePats = [
    {n:'Bullish Engulfing', bull:true,  s:'Strong'},
    {n:'Morning Star',      bull:true,  s:'Medium'},
    {n:'Hammer',            bull:true,  s:'Strong'},
    {n:'Doji',              bull:null,  s:'Neutral'},
    {n:'Shooting Star',     bull:false, s:'Weak'},
  ];
  document.getElementById('candlePatterns').innerHTML = candlePats.map(p=>{
    const c = p.bull===true?'var(--green)':p.bull===false?'var(--red)':'var(--amber)';
    return `<div class="pat-item">
      <div class="pat-icon" style="background:var(--bg4);color:${c};font-size:11px">●</div>
      <span class="pat-name">${p.n}</span>
      <span class="pat-strength" style="color:${c}">${p.s}</span>
    </div>`;
  }).join('');
}

function buildPredictionCards(sym) {
  const d = stockPrices[sym];
  if (!d) return;
  
  const horizons = [
    {lbl:'SHORT TERM', t:'1–3 Days', dir:d.chg>0?'Bullish':'Bearish', conf:Math.round(R(55,85))},
    {lbl:'WEEKLY', t:'5–7 Days', dir:R(0,1)>0.4?'Bullish':'Bearish', conf:Math.round(R(50,78))},
    {lbl:'LONG TERM', t:'1–3 Months', dir:R(0,1)>0.35?'Bullish':'Bearish', conf:Math.round(R(45,72))},
  ];
  document.getElementById('predictionCards').innerHTML = horizons.map(h=>{
    const bull = h.dir==='Bullish';
    const cls = bull?'pred-bull':'pred-bear';
    const cc = bull?'conf-bull':'conf-bear';
    const arrow = bull ? '↑ BULLISH' : '↓ BEARISH';
    const c = bull?'var(--green)':'var(--red)';
    return `<div class="pred-box ${cls}">
      <div class="pred-label">${h.lbl} · ${h.t}</div>
      <div class="pred-dir" style="color:${c}">${arrow}</div>
      <div class="pred-conf">
        <span style="font-size:10px;color:var(--muted)">Confidence</span>
        <div class="conf-bar"><div class="conf-fill ${cc}" style="width:${h.conf}%"></div></div>
        <span style="font-family:var(--mono);font-size:11px;font-weight:700;color:${c}">${h.conf}%</span>
      </div>
    </div>`;
  }).join('');
}

function buildStockNews(sym) {
  const headlines = [
    {h:`${sym} Q4 results beat estimates by 8%, revenue up 18% YoY`, s:1, t:'2h ago', src:'Economic Times'},
    {h:`Institutional buyers accumulate ${sym} ahead of earnings`, s:1, t:'4h ago', src:'Mint'},
    {h:`${sym} announces ₹2,000Cr buyback programme`, s:1, t:'6h ago', src:'BSE Filing'},
    {h:`FII net buyers in ${sym} for 5th consecutive session`, s:1, t:'1d ago', src:'Moneycontrol'},
    {h:`Analyst upgrades ${sym} target to ₹${Math.round(stockPrices[sym].price*1.18)}`, s:1, t:'1d ago', src:'CNBC-TV18'},
    {h:`${sym} faces margin pressure from rising input costs`, s:-1, t:'2d ago', src:'Business Standard'},
  ];
  document.getElementById('stockNews').innerHTML = headlines.map(n=>{
    const sc = n.s>0?'sent-pos':n.s<0?'sent-neg':'sent-neu';
    const sl = n.s>0?'POS':n.s<0?'NEG':'NEU';
    return `<div class="news-item">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:4px">
        <div class="news-head">${n.h}</div>
        <span class="news-sent ${sc}">${sl}</span>
      </div>
      <div class="news-meta">${n.src} · ${n.t}</div>
    </div>`;
  }).join('');
}

// ═══════════════════ PREDICTIONS PAGE ═══════════════════
async function buildPredictions() {
  const predictions = await fetchAPI('/predictions');
  if (!predictions) return;
  
  const metrics = await fetchAPI('/metrics/model');
  if (metrics) {
    const getVal = key => { const m = metrics.find(x=>x.key===key); return m?m.value:'--'; };
    document.getElementById('winRateStat').textContent = getVal('Win Rate');
    document.getElementById('totalPredStat').textContent = getVal('Total Predictions');
    document.getElementById('correctStat').textContent = getVal('Correct');
    document.getElementById('incorrectStat').textContent = getVal('Incorrect');
  }

  // Remove mock charts for now as there's no historical API
  const hc = document.getElementById('horizonChart');
  if(hc) hc.innerHTML = '<div style="color:var(--muted);text-align:center;padding:40px">Requires Historical DB</div>';
  
  const pc = document.getElementById('predDistChart');
  if(pc) pc.innerHTML = '<div style="color:var(--muted);text-align:center;padding:40px">Requires Historical DB</div>';

  document.getElementById('sectorWinRate').innerHTML = '<div style="color:var(--muted);font-size:11px;padding:8px">No sector data available</div>';

  buildPredTable('all');
}

async function buildPredTable(filter) {
  const predictions = await fetchAPI(`/predictions?filter=${filter}`);
  if (!predictions) return;
  
  document.getElementById('predTableBody').innerHTML = predictions.map((p,i)=>{
    const rc = p.pending?'':p.correct==='correct'?'pred-row-g':'pred-row-r';
    const status = p.pending ? '<span class="ind-sig sig-neutral">PENDING</span>'
      : p.result === 'correct' ? '<span class="ind-sig sig-buy">✓ CORRECT</span>'
      : '<span class="ind-sig sig-sell">✗ WRONG</span>';
    return `<tr class="${rc}">
      <td>${p.date}</td>
      <td><b>${p.symbol}</b></td>
      <td style="color:${p.prediction==='Bullish'?'var(--green)':'var(--red)'}">${p.prediction==='Bullish'?'↑':'↓'} ${p.prediction}</td>
      <td style="color:var(--muted2)">${p.horizon}</td>
      <td><span style="font-family:var(--mono);color:${p.confidence>75?'var(--green)':'var(--amber)'}">${p.confidence}%</span></td>
      <td style="color:var(--muted2)">${p.actual||'--'}</td>
      <td>${status}</td>
      <td><button class="why-btn" onclick="toggleWhy(this,${i})">Why?</button>
          <div class="why-box" id="why_${i}">${p.reason||'No analysis available'}</div></td>
    </tr>`;
  }).join('');
}

function toggleWhy(btn,i) {
  const box = document.getElementById('why_'+i);
  const open = box.style.display==='block';
  box.style.display = open?'none':'block';
  btn.textContent = open?'Why?':'Hide';
}

function filterPreds(f,btn) {
  predFilter=f;
  document.querySelectorAll('#view-predictions .tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  buildPredTable(f);
}

// ═══════════════════ SYSTEM HEALTH ═══════════════════
async function buildHealth() {
  const health = await fetchAPI('/health/status');
  if (health) {
    document.getElementById('healthCards').innerHTML = health.map(c=>`
      <div class="health-card">
        <div class="health-indicator">
          <div class="pulse pulse-${c.status}"></div>
          <div class="health-name">${c.name}</div>
        </div>
        <div class="health-val" style="color:${c.status==='green'?'var(--green)':c.status==='amber'?'var(--amber)':'var(--red)'}">● ${c.value}</div>
        <div style="font-size:10px;color:var(--muted);margin-top:4px">${c.message}</div>
      </div>`).join('');
  }

  const perfMetrics = await fetchAPI('/metrics/performance');
  if (perfMetrics) {
    document.getElementById('perfMetrics').innerHTML = perfMetrics.map(m=>`<div class="metric-row"><span class="metric-key">${m.key}</span>
      <span class="metric-val" style="color:${m.color}">${m.value}</span></div>`).join('');
  }

  const modelMetrics = await fetchAPI('/metrics/model');
  if (modelMetrics) {
    document.getElementById('modelMetrics').innerHTML = modelMetrics.map(m=>`<div class="metric-row"><span class="metric-key">${m.key}</span>
      <span class="metric-val" style="color:${m.color}">${m.value}</span></div>`).join('');
  }

  const logs = [];
  document.getElementById('systemLog').innerHTML = logs.length ? logs.map(l=>`
    <div class="log-line log-${l.c}">
      <span class="log-time">${l.t}</span>${l.m}
    </div>`).join('') : '<div style="color:var(--muted);font-size:11px;padding:8px">No logs available in database</div>';

  if (perfMetrics) {
    document.getElementById('orderStats').innerHTML = perfMetrics.map(m=>`<div class="metric-row"><span class="metric-key">${m.key}</span>
      <span class="metric-val" style="color:${m.color}">${m.value}</span></div>`).join('');
  } else {
    document.getElementById('orderStats').innerHTML = '<div style="color:var(--muted);font-size:11px;padding:8px">No order stats available</div>';
  }

  document.getElementById('pipelineStats').innerHTML = '<div style="color:var(--muted);font-size:11px;padding:8px">Integration pending</div>';
  document.getElementById('portfolioHealth').innerHTML = '<div style="color:var(--muted);font-size:11px;padding:8px">Portfolio connection pending</div>';
}

// ═══════════════════ VIEW ROUTER ═══════════════════
function showView(v) {
  document.querySelectorAll('.view').forEach(el=>el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(el=>el.classList.remove('active'));
  document.getElementById('view-'+v).classList.add('active');
  document.getElementById('nav-'+v).classList.add('active');

  // Close stock detail if switching away
  if(v!=='markets') {
    const det = document.getElementById('stock-detail');
    if(det && det.style.display!=='none') {
      closeStockDetail();
    }
  }

  if(v==='home') buildHome();
  if(v==='markets') { buildStocksTable(currentSector,currentSearch); }
  if(v==='predictions') buildPredictions();
  if(v==='health') buildHealth();
}

// ═══════════════════ LIVE UPDATES ═══════════════════
function liveUpdate() {
  // Randomly update 3-5 stock prices
  const keys = Object.keys(stockPrices);
  if (keys.length === 0) return;
  
  const n = Math.floor(R(3,6));
  for(let i=0;i<n;i++) {
    const k = keys[Math.floor(Math.random()*keys.length)];
    stockPrices[k].price *= (1 + R(-0.0008, 0.0008));
    stockPrices[k].chg += R(-0.05, 0.05);
  }

  // Update current view if markets
  const activeView = document.querySelector('.view.active');
  if(activeView && activeView.id==='view-markets') {
    if(document.getElementById('stock-detail').style.display!=='none' && currentStock) {
      const d = stockPrices[currentStock];
      document.getElementById('detailPrice').textContent = '₹'+fmt(d.price);
      const up = d.chg>=0;
      document.getElementById('detailChange').innerHTML =
        `<span class="${up?'up':'dn'}">${up?'+':''}${fmt(d.chg)}% (${up?'+':''}₹${fmt(d.price*d.chg/100)})</span>`;
    }
  }
}

// ═══════════════════ INIT ═══════════════════
buildTicker();
updateClock();
buildHome();
setInterval(updateClock, 1000);
setInterval(liveUpdate, 2000);
setInterval(buildTicker, 10000);
