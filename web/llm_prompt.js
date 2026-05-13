import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "comfyui.llm_prompt.generator",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "LLMPromptGenerator") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, []);
            const w = ComfyWidgets["STRING"](
                this,
                "generated_prompt",
                ["STRING", { multiline: true }],
                app
            ).widget;
            w.inputEl.readOnly = true;
            w.inputEl.placeholder = "Generated prompt appears here after Queue Prompt…";
            this.generatedPromptWidget = w;

            // Restore last value from saved workflow (set server-side via extra_pnginfo).
            const saved = this.widgets_values?.[this.widgets.length - 1];
            if (typeof saved === "string" && saved.length) w.value = saved;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, [message]);
            if (this.generatedPromptWidget && message?.text?.[0] != null) {
                this.generatedPromptWidget.value = message.text[0];
            }
        };
    },
});
