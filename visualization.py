import matplotlib.pyplot as plt
import torch

# Try importing IPython display for notebook support
try:
    from IPython.display import display, clear_output
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

class Visualizer:
    def __init__(self, dataset, device="cpu", samples_to_visualize=2000, notebook_mode=False, max_steps=100):
        self.dataset = dataset
        self.device = device
        self.samples_to_visualize = samples_to_visualize
        self.notebook_mode = notebook_mode
        self.max_steps = max_steps
        
        if self.notebook_mode and not IPYTHON_AVAILABLE:
            print("Warning: notebook_mode=True but IPython not available. Falling back to standard mode.")
            self.notebook_mode = False

        if not self.notebook_mode:
            # Enable interactive mode for scripts
            plt.ion()
        
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.ax_scatter = self.axes[0, 0]
        self.ax_loss    = self.axes[0, 1]
        self.ax_quality = self.axes[1, 0]
        self.ax_modes   = self.axes[1, 1]
        
        self._init_scatter()
        self._init_loss()
        self._init_quality()
        self._init_modes()
        
        self.fig.tight_layout()
        
        if self.notebook_mode:
            # In notebook, initial display
            clear_output(wait=True)
            display(self.fig)
        else:
            plt.draw()
            plt.pause(0.01)

    def _init_scatter(self):
        # (A) Scatter
        perm = torch.randperm(len(self.dataset))
        real_vis = self.dataset.x[perm[:self.samples_to_visualize]].cpu()
        
        self.ax_scatter.scatter(real_vis[:, 0], real_vis[:, 1], s=10, alpha=0.35, label="Real")
        
        # Initial placeholder for fake data
        self.fake_sc = self.ax_scatter.scatter([], [], s=10, alpha=0.35, label="Generated")
        
        self.ax_scatter.axis("equal")
        self.ax_scatter.legend(loc="upper right")
        self.ax_scatter.set_title("Real vs Generated")

    def _init_loss(self):
        # (B) Loss curves
        self.lossD_hist, self.lossG_hist, self.step_hist = [], [], []
        (self.lineD,) = self.ax_loss.plot([], [], label="lossD")
        (self.lineG,) = self.ax_loss.plot([], [], label="lossG")
        self.ax_loss.set_xlabel("step")
        self.ax_loss.set_ylabel("loss")
        self.ax_loss.legend(loc="upper right")
        self.ax_loss.set_title("Loss evolution")

    def _init_quality(self):
        # (C) Quality metrics with TWO Y-AXES
        # Left Y-axis: SWD
        self.swd_hist, self.step_q_hist = [], []
        (self.lineSWD,) = self.ax_quality.plot([], [], label="SWD (↓)")
        self.ax_quality.set_xlabel("step")
        self.ax_quality.set_ylabel("SWD (left axis)")
        self.ax_quality.set_title("Quality metrics evolution (two y-axes)")
        
        # Right Y-axis: JS divergence
        self.ax_quality_r = self.ax_quality.twinx()
        self.js_hist = []
        (self.lineJS,) = self.ax_quality_r.plot([], [], label="JS to uniform (↓)", color="tab:orange")
        self.ax_quality_r.set_ylabel("JS divergence (right axis)")
        
        # Combined legend for both axes
        lines_for_legend = [self.lineSWD, self.lineJS]
        labels_for_legend = [l.get_label() for l in lines_for_legend]
        self.ax_quality.legend(lines_for_legend, labels_for_legend, loc="upper right")

    def _init_modes(self):
        # (D) Mode ratios bars
        self.k = self.dataset.k
        x_modes = list(range(self.k))
        self.bars = self.ax_modes.bar(x_modes, [0.0] * self.k)
        self.ax_modes.set_xticks(x_modes)
        self.ax_modes.set_xlabel("mode")
        self.ax_modes.set_ylabel("ratio")
        self.ax_modes.set_ylim(0.0, 1.0)
        self.ax_modes.set_title("Mode ratios (generated)")

    def update(self, step, lossD, lossG, fake_samples, swd, stats):
        # 1. Update scatter plot
        self.fake_sc.set_offsets(fake_samples.cpu().numpy())
        
        # 2. Update loss curves
        self.step_hist.append(step)
        self.lossD_hist.append(lossD)
        self.lossG_hist.append(lossG)
        
        self.lineD.set_data(self.step_hist, self.lossD_hist)
        self.lineG.set_data(self.step_hist, self.lossG_hist)
        
        self.ax_loss.set_xlim(0, max(step, self.max_steps))
        self.ax_loss.set_ylim(0.0, max(max(self.lossD_hist + [1]), max(self.lossG_hist + [1])) * 1.1)
        
        # 3. Update metrics
        self.step_q_hist.append(step)
        self.swd_hist.append(swd)
        self.js_hist.append(stats['js_to_uniform'])
        
        self.lineSWD.set_data(self.step_q_hist, self.swd_hist)
        self.lineJS.set_data(self.step_q_hist, self.js_hist)
        
        self.ax_quality.set_xlim(0, max(step, self.max_steps))
        
        # simple scaling for Y
        max_val_l = max(max(self.swd_hist + [0.1]), 0)
        max_val_r = max(0, max(self.js_hist + [0.1]))
        
        self.ax_quality.set_ylim(0, max_val_l * 1.2)
        self.ax_quality_r.set_ylim(0, max_val_r * 1.2)
        
        # 4. Update mode bars
        # stats['counts'] is a tensor of counts per mode
        counts = stats['counts'].cpu().numpy()
        total = counts.sum()
        if total < 1: total = 1
        ratios = counts / total
        
        for bar, h in zip(self.bars, ratios):
            bar.set_height(h)
            
        if self.notebook_mode:
            clear_output(wait=True)
            display(self.fig)
        else:
            plt.draw()
            plt.pause(0.01)

    def show(self):
        if not self.notebook_mode:
            plt.ioff()
            plt.show()
