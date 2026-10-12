"use strict";

const $ = (id) => document.getElementById(id);
const labels = {ready: "准备执行", running: "执行中", awaiting_approval: "等待确认", awaiting_input: "等待补充", succeeded: "完成", failed: "未完成", cancelled: "已停止"};
const eventLabels = {task_created: "任务已保存", planning_started: "选择下一步工具", planning_finished: "模型返回决策", clarification_requested: "需要补充信息", clarification_answered: "已收到补充信息", approval_requested: "文件变更等待确认", approval_granted: "已确认此次变更", approval_rejected: "已拒绝变更", action_started: "执行已授权操作", action_verified: "文件终态验证通过", action_recovered: "恢复并核对已完成操作", recovery_needs_approval: "恢复后需要重新确认", recovery_conflict: "恢复遇到文件状态冲突", tool_rejected: "工具参数或边界检查未通过", tool_observed: "获得工具结果", action_failed: "文件操作失败", verification_failed: "文件终态验证失败", execution_conflict: "执行遇到文件状态冲突", task_finished: "Agent 结束本轮", task_failed: "任务未完成", task_cancelled: "任务已停止", budget_exhausted: "任务预算已用尽", cancellation_requested: "已保存停止请求", cancel_requested: "已保存停止请求"};
const toolLabels = {list_files: "查看文件", read_file: "读取文件", search_files: "搜索文本", search_content: "检索索引", write_file: "写入文件", append_file: "追加内容", copy_file: "复制文件", move_file: "移动文件", make_directory: "创建目录"};
let config = null;
let selectedId = null;
let selectedTask = null;
let parentId = null;
let taskSnapshot = "";
let listSnapshot = "";
let polling = false;
let serverOffline = false;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function notice(message) {
  $("notice").textContent = message || "";
  $("notice").hidden = !message;
}

async function api(path, body) {
  const options = {headers: {Accept: "application/json"}, cache: "no-store"};
  if (body !== undefined) {
    options.method = "POST";
    options.headers["Content-Type"] = "application/json";
    options.headers["X-CSRF-Token"] = config.csrf_token;
    options.body = JSON.stringify(body);
  }
  let response;
  try { response = await fetch(path, options); }
  catch (_) { throw new Error("工作台暂时无法连接。请确认本地服务仍在运行；重启后刷新此页面可重新连接。"); }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}

function saveSelection(id) {
  try {
    if (id) localStorage.setItem("samantha.selectedTask", id);
    else localStorage.removeItem("samantha.selectedTask");
  } catch (_) { /* The server is the persistent source of truth. */ }
}

function timeText(value, short = false) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString("zh-CN", short ? {month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit"} : {hour: "2-digit", minute: "2-digit", second: "2-digit"});
}

function statusText(task) {
  if (task.cancel_requested && !["cancelled", "succeeded", "failed"].includes(task.status)) return "正在停止";
  if (task.runner_active && task.status === "ready") return "已排队";
  if (!task.runner_active && task.status === "running") return "可恢复";
  return labels[task.status] || task.status;
}

function renderList(tasks) {
  const signature = JSON.stringify([tasks, selectedId]);
  if (signature === listSnapshot) return;
  listSnapshot = signature;
  $("task-count").textContent = tasks.length;
  const list = $("task-list");
  list.replaceChildren();
  if (!tasks.length) list.append(element("p", "sidebar-empty", "暂无任务。从右侧描述一个目标开始。"));
  for (const task of tasks) {
    const button = element("button", "task-item" + (task.id === selectedId ? " selected" : ""));
    button.type = "button";
    button.setAttribute("aria-pressed", String(task.id === selectedId));
    button.title = task.request;
    button.append(element("div", "task-item-title", task.request));
    const meta = element("div", "task-item-meta");
    meta.append(element("span", "task-item-status", statusText(task)), element("span", "", timeText(task.created_at, true)));
    button.append(meta);
    button.addEventListener("click", () => selectTask(task.id));
    list.append(button);
  }
}

async function selectTask(id) {
  selectedId = id;
  taskSnapshot = "";
  $("task-detail").hidden = true;
  saveSelection(id);
  parentId = null;
  $("followup-label").hidden = true;
  notice("");
  await refresh();
}

function clearTask() {
  selectedId = null;
  selectedTask = null;
  taskSnapshot = "";
  parentId = null;
  saveSelection(null);
  $("task-detail").hidden = true;
  $("empty-state").hidden = false;
  $("followup-label").hidden = true;
  $("request").value = "";
  $("request").focus();
  notice("");
  refresh();
}

function codeBlock(value) {
  return element("pre", "code-block", typeof value === "string" ? value : JSON.stringify(value, null, 2));
}

function inspectDetails(label, value) {
  const details = element("details");
  details.append(element("summary", "", label), codeBlock(value));
  return details;
}

function actionButton(label, className, operation) {
  const button = element("button", "button " + className, label);
  button.type = "button";
  button.addEventListener("click", async () => {
    button.disabled = true;
    try { await operation(); }
    catch (error) { notice(error.message); }
    finally { button.disabled = false; }
  });
  return button;
}

async function mutate(action, body = {}, taskId = selectedId) {
  notice("");
  const task = await api(`/api/tasks/${encodeURIComponent(taskId)}/${action}`, body);
  if (selectedId === taskId) renderTask(task);
  await refresh();
}

function renderApproval(task) {
  const box = $("approval");
  const action = task.pending_action;
  box.hidden = task.status !== "awaiting_approval" || !action;
  if (box.hidden) return;
  box.replaceChildren();
  const heading = element("div", "approval-heading");
  heading.append(element("span", "", "!"), element("h2", "", "确认这次文件变更"));
  box.append(heading, element("p", "panel-copy", action.reason || "请核对路径与内容。确认仅适用于当前展示的操作，文件状态发生变化时会重新检查。"));
  box.append(codeBlock(action.preview || {tool: action.tool, args: action.args}));
  box.append(inspectDetails("查看参数与文件状态", {tool: action.tool, args: action.args, before: action.before, expected: action.expected}));
  const buttons = element("div", "action-buttons");
  buttons.append(actionButton("拒绝并停止", "secondary", () => mutate("approve", {action_id: action.id, approved: false}, task.id)), actionButton("确认并执行", "primary", () => mutate("approve", {action_id: action.id, approved: true}, task.id)));
  box.append(buttons);
}

function renderClarification(task) {
  const box = $("clarification");
  box.hidden = task.status !== "awaiting_input";
  if (box.hidden) return;
  const savedAnswer = box.dataset.task === task.id ? box.querySelector("textarea")?.value || "" : "";
  box.dataset.task = task.id;
  box.replaceChildren(element("h2", "", "补充一点信息，继续任务"), element("p", "panel-copy", task.question));
  const form = element("form");
  const input = element("textarea");
  input.setAttribute("aria-label", "补充信息");
  input.placeholder = "输入你的回答…";
  input.maxLength = 12000;
  input.required = true;
  input.value = savedAnswer;
  const buttons = element("div", "action-buttons");
  const submit = element("button", "button primary", "提交并继续");
  submit.type = "submit";
  buttons.append(submit);
  form.append(input, buttons);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!input.value.trim()) return;
    submit.disabled = true;
    try { await mutate("answer", {text: input.value}, task.id); }
    catch (error) { notice(error.message); }
    finally { submit.disabled = false; }
  });
  box.append(form);
}

function renderTimeline(task) {
  const list = $("timeline");
  const openDetails = new Set(Array.from(list.querySelectorAll("details[open]")).map((node) => node.dataset.event));
  list.replaceChildren();
  const actions = new Map((task.actions || []).map((action) => [action.id, action]));
  for (const event of task.events || []) {
    const payload = event.payload || {};
    const action = actions.get(payload.action_id);
    const verified = ["action_verified", "action_recovered"].includes(event.kind) && action?.verification?.ok === true;
    const failed = /failed|conflict|rejected|exhausted/.test(event.kind) || payload.status === "failed";
    const item = element("li");
    item.append(element("span", "timeline-dot" + (verified ? " verified" : failed ? " error" : "")));
    const line = element("div", "event-line");
    line.append(element("span", "event-title", eventLabels[event.kind] || event.kind));
    if (verified) line.append(element("span", "verified-badge", "已验证"));
    line.append(element("time", "event-time", timeText(event.created_at)));
    item.append(line);
    const tool = payload.tool || action?.tool;
    const detail = payload.error || payload.question || (tool ? `${toolLabels[tool] || tool} · ${tool}` : payload.step ? `第 ${payload.step} 步` : "");
    if (detail) item.append(element("p", "event-detail", detail));
    const evidence = action && ["approval_requested", "tool_observed", "action_verified", "action_recovered"].includes(event.kind) ? {tool: action.tool, args: action.args, result: action.result, verification: action.verification, ...payload} : payload;
    const details = inspectDetails(verified ? "查看验证证据" : "查看记录", evidence);
    details.dataset.event = String(event.id);
    details.open = openDetails.has(String(event.id));
    item.append(details);
    list.append(item);
  }
}

function renderTask(task) {
  if (task.id !== selectedId) return;
  const signature = JSON.stringify(task);
  if (signature === taskSnapshot) return;
  taskSnapshot = signature;
  selectedTask = task;
  $("empty-state").hidden = true;
  $("task-detail").hidden = false;
  $("task-id").textContent = `TASK / ${task.id.slice(0, 12).toUpperCase()}`;
  $("task-title").textContent = task.request;
  $("task-status").className = "status-pill " + (labels[task.status] ? task.status : "");
  $("task-status").textContent = statusText(task);
  const metrics = $("metrics");
  metrics.replaceChildren();
  for (const [value, label] of [[`${task.steps || 0} / ${task.max_steps}`, "决策步数"], [task.model_calls || 0, "实际模型请求"], [(task.actions || []).filter((action) => action.status === "verified" && action.verification?.ok === true).length, "已验证文件操作"]]) {
    const metric = element("div", "metric");
    metric.append(element("strong", "", value), element("span", "", label));
    metrics.append(metric);
  }
  let error = task.runner_error || (task.status === "failed" ? task.error : "");
  if (task.cancel_requested && !["cancelled", "succeeded", "failed"].includes(task.status)) error = "停止请求已保存，等待当前步骤结束。已完成的文件变更会保留。";
  else if (!task.runner_active && task.status === "running" && !error) error = "任务停留在上次的执行位置。点击「恢复执行」，核对文件状态后继续。";
  $("task-error").textContent = error || "";
  $("task-error").hidden = !error;
  renderApproval(task);
  renderClarification(task);
  $("result").hidden = !task.answer;
  $("result").replaceChildren(element("h2", "", "本轮答复"), element("p", "result-copy", task.answer || ""));
  const terminal = ["succeeded", "failed", "cancelled"].includes(task.status);
  $("resume").hidden = terminal || task.runner_active || ["awaiting_approval", "awaiting_input"].includes(task.status);
  $("resume").textContent = "恢复执行";
  $("followup").hidden = !terminal;
  $("cancel").hidden = terminal;
  $("cancel").disabled = Boolean(task.cancel_requested);
  $("cancel").textContent = task.cancel_requested ? "正在停止…" : "停止任务";
  renderTimeline(task);
}

async function refresh() {
  if (!config || polling) return;
  polling = true;
  const id = selectedId;
  try {
    const data = await api("/api/tasks");
    renderList(data.tasks);
    if (id && selectedId === id) {
      if (data.tasks.some((task) => task.id === id)) {
        const task = await api(`/api/tasks/${encodeURIComponent(id)}`);
        if (selectedId === id) renderTask(task);
      } else {
        selectedId = null;
        selectedTask = null;
        saveSelection(null);
        $("task-detail").hidden = true;
        $("empty-state").hidden = false;
      }
    }
    if (serverOffline) { notice(""); serverOffline = false; }
  } catch (error) { serverOffline = true; notice(error.message); }
  finally { polling = false; }
}

$("new-task").addEventListener("click", clearTask);
for (const example of document.querySelectorAll("[data-example]")) {
  example.addEventListener("click", () => { $("request").value = example.dataset.example; $("request").focus(); });
}
$("task-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const request = $("request").value.trim();
  if (!request || !config) return;
  $("submit-task").disabled = true;
  notice("");
  try {
    const body = {request, max_steps: Number($("max-steps").value)};
    if (parentId) {
      body.parent_id = parentId;
      if (selectedTask?.id === parentId) body.workspace = selectedTask.workspace;
    }
    const task = await api("/api/tasks", body);
    selectedId = task.id;
    parentId = null;
    saveSelection(task.id);
    $("followup-label").hidden = true;
    $("request").value = "";
    renderTask(task);
    await refresh();
  } catch (error) { notice(error.message); }
  finally { $("submit-task").disabled = false; }
});
$("resume").addEventListener("click", async () => {
  $("resume").disabled = true;
  try { await mutate("run"); } catch (error) { notice(error.message); }
  finally { $("resume").disabled = false; }
});
$("cancel").addEventListener("click", async () => {
  $("cancel").disabled = true;
  try { await mutate("cancel"); } catch (error) { notice(error.message); }
  finally { $("cancel").disabled = Boolean(selectedTask?.cancel_requested); }
});
$("followup").addEventListener("click", () => {
  parentId = selectedId;
  $("followup-label").textContent = `接续任务 ${selectedId.slice(0, 8)} · 引用已验证操作`;
  $("followup-label").hidden = false;
  $("request").focus();
  $("request").scrollIntoView({behavior: "smooth", block: "center"});
});

async function boot() {
  try {
    config = await api("/api/config");
    $("workspace-path").textContent = config.workspace;
    $("workspace-path").title = config.workspace;
    $("model-name").textContent = config.model || "未配置模型";
    $("model-status").textContent = config.model_configured ? "待实际请求验证连接" : "请检查服务端模型配置";
    $("submit-task").disabled = false;
    try { selectedId = localStorage.getItem("samantha.selectedTask"); } catch (_) { /* Optional browser preference. */ }
    await refresh();
    setInterval(() => { if (!document.hidden) refresh(); }, 1500);
    document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
  } catch (error) { notice(error.message); }
}
boot();
