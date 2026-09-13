(() => {
  "use strict";
  const $ = (selector) => document.querySelector(selector);
  const arena = $("#arena");
  const activity = $("#activity");
  const arenaContext = arena.getContext("2d");
  const activityContext = activity.getContext("2d");
  const token = document.querySelector('meta[name="flytype-control-token"]').content;
  let latest = null;
  let lastActivity = null;
  let displayWorld = null;
  let priorWorld = null;
  let transitionStart = 0;
  let shownRevision = -1;
  const transitionDuration = 260;
  const colors = {arena:"#05090b",grid:"rgba(84,113,123,.11)",boundary:"rgba(86,219,196,.22)",brick:"#70a9ff",brickEdge:"rgba(193,220,255,.58)",paddle:"#56dbc4",paddleGlow:"rgba(86,219,196,.26)",ball:"#ffc45b",ballGlow:"rgba(255,196,91,.3)"};

  function sizeCanvas(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width === width && canvas.height === height) return false;
    canvas.width = width; canvas.height = height; return true;
  }
  const mix = (a, b, amount) => a + (b - a) * amount;
  function interpolatedWorld(now) {
    if (!latest || !latest.world) return null;
    if (!priorWorld) return latest.world;
    const t = Math.min(1, Math.max(0, (now - transitionStart) / transitionDuration));
    const smooth = t * t * (3 - 2 * t);
    return {...latest.world,paddle_x:mix(priorWorld.paddle_x,latest.world.paddle_x,smooth),ball_x:mix(priorWorld.ball_x,latest.world.ball_x,smooth),ball_y:mix(priorWorld.ball_y,latest.world.ball_y,smooth)};
  }

  function drawArena(now) {
    sizeCanvas(arena);
    // Resizing a canvas clears it, so the spike field has to be repainted from
    // the last committed observation rather than waiting for the next one.
    if (sizeCanvas(activity) && lastActivity) drawActivity(lastActivity.neural, lastActivity.source);
    const width = arena.width, height = arena.height;
    const world = interpolatedWorld(now), geometry = latest && latest.geometry;
    arenaContext.clearRect(0,0,width,height); arenaContext.fillStyle=colors.arena; arenaContext.fillRect(0,0,width,height);
    if (!world || !geometry) {
      arenaContext.fillStyle="#52616b"; arenaContext.font=`${Math.max(13,height*.025)}px ui-monospace,monospace`; arenaContext.textAlign="center"; arenaContext.fillText("WAITING FOR COMMITTED STATE",width/2,height/2); requestAnimationFrame(drawArena); return;
    }
    // Frame the playfield, not the sensory canvas. The 320x180 frame the model
    // sees carries a HUD strip above field_top and a caption strip below
    // field_bottom, neither of which is part of the game; drawing all 180 rows
    // here left ~40% of the shell as dead space under the paddle.
    const logicalWidth=geometry.width||320;
    const top=Math.max(0,(geometry.field_top??0)-4), bottom=Math.min(180,(geometry.field_bottom??180)+5);
    const logicalHeight=Math.max(1,bottom-top);
    const scale=Math.min(width/logicalWidth,height/logicalHeight), ox=(width-logicalWidth*scale)/2, oy=(height-logicalHeight*scale)/2;
    const X=(x)=>ox+x*scale, Y=(y)=>oy+(y-top)*scale;
    arenaContext.save(); arenaContext.beginPath(); arenaContext.rect(ox,oy,logicalWidth*scale,logicalHeight*scale); arenaContext.clip();
    arenaContext.strokeStyle=colors.grid; arenaContext.lineWidth=1;
    for(let x=0;x<=logicalWidth;x+=20){arenaContext.beginPath();arenaContext.moveTo(X(x),oy);arenaContext.lineTo(X(x),oy+logicalHeight*scale);arenaContext.stroke()}
    for(let y=Math.ceil(top/20)*20;y<=bottom;y+=20){arenaContext.beginPath();arenaContext.moveTo(ox,Y(y));arenaContext.lineTo(ox+logicalWidth*scale,Y(y));arenaContext.stroke()}
    arenaContext.strokeStyle=colors.boundary; arenaContext.setLineDash([5*scale,5*scale]);
    arenaContext.beginPath();arenaContext.moveTo(ox,Y(geometry.field_top));arenaContext.lineTo(ox+logicalWidth*scale,Y(geometry.field_top));arenaContext.stroke();
    arenaContext.beginPath();arenaContext.moveTo(ox,Y(geometry.field_bottom));arenaContext.lineTo(ox+logicalWidth*scale,Y(geometry.field_bottom));arenaContext.stroke();arenaContext.setLineDash([]);
    const columns=world.columns||1,rows=world.rows||1,gap=geometry.brick_gap,margin=geometry.field_margin;
    const brickWidth=(logicalWidth-2*margin-(columns-1)*gap)/columns,bits=world.bricks||"";
    for(let index=0;index<rows*columns;index+=1){if(bits[index]!=="1")continue;const row=Math.floor(index/columns),column=index%columns,x=margin+column*(brickWidth+gap),y=geometry.brick_top+row*(geometry.brick_height+gap);arenaContext.fillStyle=colors.brick;arenaContext.fillRect(X(x),Y(y),Math.max(1,brickWidth*scale),Math.max(1,geometry.brick_height*scale));arenaContext.fillStyle=colors.brickEdge;arenaContext.fillRect(X(x),Y(y),Math.max(1,brickWidth*scale),Math.max(1,scale*.65))}
    arenaContext.shadowColor=colors.paddleGlow;arenaContext.shadowBlur=13*scale;arenaContext.fillStyle=colors.paddle;arenaContext.fillRect(X(world.paddle_x),Y(geometry.paddle_top),world.paddle_width*scale,geometry.paddle_height*scale);arenaContext.shadowBlur=0;arenaContext.fillStyle="rgba(255,255,255,.45)";arenaContext.fillRect(X(world.paddle_x),Y(geometry.paddle_top),world.paddle_width*scale,Math.max(1,scale));
    const diameter=world.ball_diameter||geometry.ball_diameter,radius=diameter*scale/2;
    arenaContext.shadowColor=colors.ballGlow;arenaContext.shadowBlur=15*scale;arenaContext.fillStyle=colors.ball;arenaContext.beginPath();arenaContext.arc(X(world.ball_x+diameter/2),Y(world.ball_y+diameter/2),radius,0,Math.PI*2);arenaContext.fill();arenaContext.shadowBlur=0;arenaContext.fillStyle="rgba(255,255,255,.72)";arenaContext.beginPath();arenaContext.arc(X(world.ball_x+diameter*.37),Y(world.ball_y+diameter*.34),Math.max(1,radius*.18),0,Math.PI*2);arenaContext.fill();arenaContext.restore();
    displayWorld=world; requestAnimationFrame(drawArena);
  }
  function numeric(value,digits=1){return Number.isFinite(Number(value))?Number(value).toFixed(digits):"—"}
  function drawActivity(neural,source){
    lastActivity={neural,source};
    sizeCanvas(activity);const width=activity.width,height=activity.height;activityContext.fillStyle="#070c0e";activityContext.fillRect(0,0,width,height);
    const buckets=neural&&Array.isArray(neural.spike_buckets)?neural.spike_buckets:[],values=buckets.flat?buckets.flat().map(Number).filter(Number.isFinite):[];
    if(!values.length){const label=source==="fixture"?"SYNTHETIC FIXTURE — NO CNS SPIKE FIELD":"NO SPIKE BINS";activityContext.fillStyle=source==="fixture"?"#ffc45b":"#52616b";activityContext.textAlign="center";
      // Fit the caption to the narrower axis; a tall panel must not scale the
      // text past the panel's own width.
      let size=Math.max(10,Math.min(height*.16,width*.055));activityContext.font=`${size}px ui-monospace,monospace`;
      while(size>9&&activityContext.measureText(label).width>width*.92){size-=1;activityContext.font=`${size}px ui-monospace,monospace`}
      activityContext.fillText(label,width/2,height/2+size*.35);return}
    const maximum=Math.max(1,...values),count=values.length,columns=Math.min(64,count),rows=Math.ceil(count/columns),cellWidth=width/columns,cellHeight=height/rows;
    values.forEach((value,index)=>{const intensity=Math.sqrt(Math.max(0,value)/maximum);activityContext.fillStyle=`rgba(86,219,196,${.09+intensity*.91})`;activityContext.fillRect((index%columns)*cellWidth,Math.floor(index/columns)*cellHeight,Math.max(1,cellWidth-1),Math.max(1,cellHeight-1))});
  }
  function setControl(control){const value=Math.max(-1,Math.min(1,Number(control)||0));$("#control").textContent=`${value>=0?"+":""}${value.toFixed(3)}`;$("#control-sign").textContent=value<-.02?"←":value>.02?"→":"↔";const marker=$("#control-marker"),bar=$("#control-bar");marker.style.left=`${50+value*50}%`;bar.style.width=`${Math.abs(value)*50}%`;bar.classList.toggle("left",value<0)}
  // How well this frozen decoder decoded held-out probes, against its own
  // shuffled-label control and against a decoder that learned nothing. Shown
  // permanently: an offset readout without it invites the viewer to read
  // steering into what may be noise.
  function setDecoderQuality(quality,isFixture){
    const node=$("#decoder-quality");
    if(isFixture){node.textContent="Fixture mode: the control is scripted from the ball position, not decoded.";node.className="truth-note synthetic";return}
    if(!quality||!Number.isFinite(Number(quality.validation_mae_px))){node.textContent="Frozen decoder. Runtime input is neural firing only.";node.className="truth-note";return}
    const mae=Number(quality.validation_mae_px),shuffled=Number(quality.shuffled_control_mae_px),blind=Number(quality.predict_the_mean_mae_px);
    const range=Array.isArray(quality.offset_range_px)?`${Math.round(quality.offset_range_px[1])}`:"?";
    // "Beats its controls" means beating BOTH the shuffled labels and a
    // decoder that always predicts the mean, by a margin worth reporting.
    const informative=Number.isFinite(shuffled)&&Number.isFinite(blind)&&mae<shuffled*0.9&&mae<blind*0.9;
    // Fitted frozen and unreinforced, run plastic and reinforced: the rate
    // distribution the z-scoring assumes is not the one it is being fed.
    const regime=quality.calibrated_frozen&&!quality.run_frozen
      ? " Calibrated frozen and unreinforced but run plastic and reinforced, so the fitted normalisation does not match these rates."
      : "";
    node.className=`truth-note ${informative?"":"warn"}`;
    node.textContent=(informative
      ?`Frozen decoder; runtime input is neural firing only. Held-out probe error ${mae.toFixed(0)} px over ±${range} px, against ${shuffled.toFixed(0)} px for shuffled labels and ${blind.toFixed(0)} px for predicting the mean.`
      :`Frozen decoder; input is neural firing only. Held-out error ${mae.toFixed(0)} px over ±${range} px — no better than shuffled labels (${shuffled.toFixed(0)} px) or predicting the mean (${blind.toFixed(0)} px). Not a demonstrated read of ball position; the paddle is not shown to be steering.`)+regime;
  }
  function setRunnerState(state){
    const status=state.runner_status||"starting";$("#runner-status").textContent=status.toUpperCase();$("#compute-label").textContent=status==="computing"||status==="pausing"?"PROPAGATING":status.toUpperCase();const dot=$("#status-dot");dot.className=["running","computing"].includes(status)?"live":status;
    const paused=status==="paused"||status==="pausing";$("#pause-label").textContent=paused?"RESUME":"PAUSE AFTER CURRENT";$("#pause-icon").textContent=paused?"▶":"Ⅱ";$("#pause").dataset.action=paused?"resume":"pause";$("#pause").disabled=["error","stopped"].includes(status);$("#stop").disabled=status==="stopped";const error=state.runner_error;$("#error").textContent=error?`${error.type}: ${error.reason}`:"";
  }
  function render(state){
    setRunnerState(state);$("#observation").textContent=String(state.observation??0).padStart(6,"0");if(!state.world||state.revision===shownRevision)return;shownRevision=state.revision;
    const isFixture=state.source==="fixture",source=$("#source");source.textContent=isFixture?"FIXTURE / SYNTHETIC":"MALECNS v1.0 / LIVE";source.className=`source-badge ${isFixture?"synthetic":""}`;$("#score").textContent=String(state.world.score||0).padStart(6,"0");$("#episode").textContent=String((state.episode||0)+1).padStart(3,"0");$("#bricks").textContent=`${state.world.bricks_left} / ${state.world.rows*state.world.columns}`;$("#lives").textContent="●".repeat(Math.max(0,state.world.lives||0))||"—";$("#action").textContent=`${state.action||"HOLD"} ${numeric(Math.abs(state.motor&&state.motor.control),2)}`;$("#cell-count").textContent=isFixture?"NO CNS":`${state.motor.cell_count} CELLS`;$("#offset").textContent=numeric(state.motor.offset_px,1);$("#motion").textContent=numeric(state.motor.relative_motion_px,1);setControl(state.motor.control);
    const event=state.event||"committed",toast=$("#event-toast");toast.textContent=event.toUpperCase();toast.className=`event-toast ${["caught","brick"].includes(event)?"impact":event==="lost"?"lost":""}`;$("#causal-label").textContent=`Observation ${state.observation} committed · ${event.replaceAll("_"," ")} · next state ${state.runner_status}`;if(state.retinal_png)$("#retinal").src=state.retinal_png;
    setDecoderQuality(state.decoder_quality,isFixture);
    $("#sensory-note").textContent=state.geometry&&state.geometry.egocentric?"The exact frame the model receives, paddle-centred: where the ball sits in it is its offset. Bricks are not drawn — the model never sees the wall.":"The exact frame the model receives, in world view. The arena above is a presentation of the same state, not the model's input.";
    const neural=state.neural||{},left=neural.left_hz,right=neural.right_hz,difference=neural.difference_hz;$("#left-rate").textContent=`${numeric(left,2)} Hz`;$("#right-rate").textContent=`${numeric(right,2)} Hz`;$("#rate-diff").textContent=`${Number(difference)>=0?"+":""}${numeric(difference,2)} Hz`;drawActivity(neural,state.source);
    priorWorld=displayWorld||(latest&&latest.world)||state.world;latest=state;transitionStart=performance.now();
  }
  async function sendControl(action){$("#error").textContent="";try{const response=await fetch("/api/control",{method:"POST",headers:{"Content-Type":"application/json","X-FlyType-Token":token},body:JSON.stringify({action})}),payload=await response.json();if(!response.ok)throw new Error(payload.error||`HTTP ${response.status}`);render(payload.state)}catch(error){$("#error").textContent=`Control failed: ${error.message}`}}
  $("#pause").addEventListener("click",()=>sendControl($("#pause").dataset.action||"pause"));$("#stop").addEventListener("click",()=>sendControl("stop"));// Both canvases are resized inside the animation loop, which repaints the
// spike field afterwards; resizing it here would blank it until the next
// observation landed.
window.addEventListener("resize",()=>{sizeCanvas(arena)});
  const refresh=()=>fetch("/api/state").then((response)=>response.json()).then(render).catch(()=>{});refresh();setInterval(refresh,750);const events=new EventSource(`/api/events?token=${encodeURIComponent(token)}`);events.onmessage=({data})=>render(JSON.parse(data));events.onerror=()=>{if(!latest||!["stopped","error"].includes(latest.runner_status))refresh()};requestAnimationFrame(drawArena);
})();
