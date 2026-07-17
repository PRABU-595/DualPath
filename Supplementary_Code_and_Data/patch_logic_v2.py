import re

with open('model/dualpath_vlm_fixed.py', 'r') as f:
    content = f.read()

# Fix the execution order in forward()
bad_forward = """        # 1. Forward Pass through backbone and capture hidden states/attention
        if is_text_only:
            outputs = self.backbone(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=True,
                **kwargs
            )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                output_attentions=True,
                **kwargs
            )
        
        # 2. Update Signals and Tracking
        context_start, context_end = self._get_context_token_range(input_ids, labels, is_text_only)
        self._reset_pathway_states(context_start, context_end)

        # Capture signals from attention hooks
        f_signal = self.monitor.get_fabrication_signal()
        raw_drift = self.monitor.get_drift_signal()
        
        # Formulate formal drift signal via tracker
        d_signal = self.drift_tracker.update(raw_drift)

        # 3. DualPath Pathway Processing
        # h_current: last hidden state [batch, seq_len, hidden_dim]
        h_current = outputs.hidden_states[-1]"""

good_forward = """        # 1. Initialize Context and Monitors FIRST so hooks capture attention!
        context_start, context_end = self._get_context_token_range(input_ids, labels, is_text_only)
        self._reset_pathway_states(context_start, context_end)

        # 2. Forward Pass through backbone and capture hidden states/attention
        if is_text_only:
            outputs = self.backbone(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=True,
                **kwargs
            )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                output_attentions=True,
                **kwargs
            )

        # h_current: last hidden state [batch, seq_len, hidden_dim]
        h_current = outputs.hidden_states[-1]

        # 3. Capture signals from attention hooks and enforce device/dtype
        f_signal = self.monitor.get_fabrication_signal().to(self.target_dtype).to(h_current.device)
        raw_drift = self.monitor.get_drift_signal().to(self.target_dtype).to(h_current.device)
        
        # Formulate formal drift signal via tracker
        d_signal = self.drift_tracker.update(raw_drift).to(self.target_dtype).to(h_current.device)

        # 4. DualPath Pathway Processing"""

content = content.replace(bad_forward, good_forward)

# Also fix generate() just to be absolutely safe on devices
bad_generate = """            h_current = out.hidden_states[-1][:, -1:, :].to(self.target_dtype)
            f_signal = self.monitor.get_fabrication_signal().to(self.target_dtype)
            raw_drift = self.monitor.get_drift_signal().to(self.target_dtype)
            d_signal = self.drift_tracker.update(raw_drift).to(self.target_dtype)
            context_embeddings = context_embeddings.to(self.target_dtype)"""

good_generate = """            h_current = out.hidden_states[-1][:, -1:, :].to(self.target_dtype)
            f_signal = self.monitor.get_fabrication_signal().to(self.target_dtype).to(h_current.device)
            raw_drift = self.monitor.get_drift_signal().to(self.target_dtype).to(h_current.device)
            d_signal = self.drift_tracker.update(raw_drift).to(self.target_dtype).to(h_current.device)
            context_embeddings = context_embeddings.to(self.target_dtype).to(h_current.device)"""

content = content.replace(bad_generate, good_generate)

with open('model/dualpath_vlm_fixed_v2.py', 'w') as f:
    f.write(content)
print('Done v2')
