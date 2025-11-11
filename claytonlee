import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from ipywidgets import FloatSlider, IntSlider, VBox, HBox, Button, Text
import random

class ThermoChargeWithTimeline:
    def __init__(self, num_sources=3, base_temp=30, fluctuation=10, efficiency=0.05,
                 battery_capacity=100, env_heat=0, time_steps=300):
        self.num_sources = num_sources
        self.base_temp = base_temp
        self.fluctuation = fluctuation
        self.efficiency = efficiency
        self.battery_capacity = battery_capacity
        self.env_heat = env_heat
        self.time_steps = time_steps
        self.t = 0

        # Scenarios
        self.scenarios = {}
        self.colors = {}
        self.battery_charge_scenarios = {}
        self.current_drawing_scenario = None
        self.prediction_texts = {}
        self.timeline_bars = {}

        # Initialize default scenario
        self.add_scenario("Scenario 1")

        # Temperature sources
        self.init_sources()
        self.voltage_generated_sources = [np.zeros(self.time_steps) for _ in range(self.num_sources)]

        # Figure
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3,1, gridspec_kw={'height_ratios':[2,2,0.5]})
        self.fig.canvas.header_visible = False

        # Heatmap
        self.heatmap_data = np.zeros((self.num_sources, self.time_steps))
        self.heatmap = self.ax1.imshow(self.heatmap_data, aspect='auto', cmap='hot', origin='lower')
        self.ax1.set_xlabel("Time Steps")
        self.ax1.set_ylabel("Heat Sources")
        self.ax1.set_title("Heat Contribution per Source")
        self.ax1.set_yticks(range(self.num_sources))
        self.ax1.set_yticklabels([f"Source {i+1}" for i in range(self.num_sources)])
        self.cbar = self.fig.colorbar(self.heatmap, ax=self.ax1)
        self.cbar.set_label("Voltage Contribution (V)")

        # Battery charge & load lines
        self.lines_charge = {}
        self.lines_load = {}
        for name in self.scenarios:
            color = self.colors[name]
            self.lines_charge[name], = self.ax2.plot([], [], color=color, label=f"{name} Charge")
            self.lines_load[name], = self.ax2.plot(range(self.time_steps), self.scenarios[name], color=color, linestyle='--', label=f"{name} Load")
            self.prediction_texts[name] = self.ax2.text(0, self.battery_capacity*1.05, "", color=color)
            # Timeline bar
            self.timeline_bars[name] = self.ax3.barh(0, 0, color=color, height=0.5, align='center')
        self.ax2.set_xlim(0, self.time_steps)
        self.ax2.set_ylim(0, self.battery_capacity*1.2)
        self.ax2.set_xlabel("Time Steps")
        self.ax2.set_ylabel("Value")
        self.ax2.set_title("Battery Charge & Load Profiles with Predictions")
        self.ax2.legend()
        self.ax2.grid(True)

        self.ax3.set_xlim(0, self.time_steps)
        self.ax3.set_ylim(-0.5, 0.5)
        self.ax3.axis('off')
        self.ax3.set_title("Battery Timeline (Full -> Empty)")

        # Scenario controls
        self.text_scenario_name = Text(value=f"Scenario {len(self.scenarios)+1}", description="New Scenario:")
        self.button_add_scenario = Button(description="Add Scenario", button_style='success')
        self.button_add_scenario.on_click(self.add_scenario_button)
        display(HBox([self.text_scenario_name, self.button_add_scenario]))

        # Sliders
        self.slider_num_sources = IntSlider(value=self.num_sources, min=1, max=5, step=1, description='Sources')
        self.slider_base_temp = IntSlider(value=self.base_temp, min=0, max=100, step=1, description='Base Temp')
        self.slider_fluctuation = IntSlider(value=self.fluctuation, min=0, max=50, step=1, description='Fluctuation')
        self.slider_efficiency = FloatSlider(value=self.efficiency, min=0.01, max=0.2, step=0.01, description='Efficiency')
        self.slider_battery_capacity = IntSlider(value=self.battery_capacity, min=10, max=200, step=10, description='Battery Cap')
        self.slider_env_heat = FloatSlider(value=self.env_heat, min=0, max=50, step=0.5, description='Env Heat')
        for s in [self.slider_num_sources, self.slider_base_temp, self.slider_fluctuation,
                  self.slider_efficiency, self.slider_battery_capacity, self.slider_env_heat]:
            s.observe(self.update_parameters, names='value')
        display(VBox([HBox([self.slider_num_sources, self.slider_base_temp, self.slider_fluctuation,
                            self.slider_efficiency, self.slider_battery_capacity, self.slider_env_heat])]))

        # Interactive drawing
        self.drawing = False
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        # Start animation
        self.ani = FuncAnimation(self.fig, self.animate, interval=50, blit=True)
        plt.show()

    def init_sources(self):
        self.base_temps = np.linspace(self.base_temp, self.base_temp + 5*(self.num_sources-1), self.num_sources)
        self.fluctuations = np.linspace(self.fluctuation, max(self.fluctuation-2*(self.num_sources-1),0), self.num_sources)
        self.efficiencies = np.linspace(self.efficiency, max(self.efficiency-0.01*(self.num_sources-1),0.01), self.num_sources)
        self.temperature_sources = [
            self.base_temps[i] + self.fluctuations[i]*np.sin(np.linspace(0, 4*np.pi, self.time_steps)+i*np.pi/4)
            for i in range(self.num_sources)
        ]

    def update_parameters(self, change):
        self.num_sources = self.slider_num_sources.value
        self.base_temp = self.slider_base_temp.value
        self.fluctuation = self.slider_fluctuation.value
        self.efficiency = self.slider_efficiency.value
        self.battery_capacity = self.slider_battery_capacity.value
        self.env_heat = self.slider_env_heat.value
        self.init_sources()
        self.ax2.set_ylim(0, self.battery_capacity*1.2)
        self.ax1.set_yticks(range(self.num_sources))
        self.ax1.set_yticklabels([f"Source {i+1}" for i in range(self.num_sources)])

    def add_scenario(self, name, load_profile=None):
        if load_profile is None:
            load_profile = np.zeros(self.time_steps)
        color = (random.random(), random.random(), random.random())
        self.scenarios[name] = load_profile
        self.colors[name] = color
        self.battery_charge_scenarios[name] = 0
        self.current_drawing_scenario = name
        self.lines_charge[name], = self.ax2.plot([], [], color=color, label=f"{name} Charge")
        self.lines_load[name], = self.ax2.plot(range(self.time_steps), load_profile, color=color, linestyle='--', label=f"{name} Load")
        self.prediction_texts[name] = self.ax2.text(0, self.battery_capacity*1.05, "", color=color)
        self.timeline_bars[name] = self.ax3.barh(0, 0, color=color, height=0.5, align='center')
        self.ax2.legend()

    def add_scenario_button(self, b):
        name = self.text_scenario_name.value
        self.add_scenario(name)
        print(f"Added scenario '{name}' for drawing.")

    # Drawing handlers
    def on_press(self, event):
        if event.inaxes == self.ax2 and self.current_drawing_scenario is not None:
            self.drawing = True
            self.update_load(event)

    def on_motion(self, event):
        if self.drawing and event.inaxes == self.ax2 and self.current_drawing_scenario is not None:
            self.update_load(event)

    def on_release(self, event):
        self.drawing = False

    def update_load(self, event):
        idx = int(event.xdata)
        if 0 <= idx < self.time_steps:
            self.scenarios[self.current_drawing_scenario][idx] = max(event.ydata, 0)
            self.lines_load[self.current_drawing_scenario].set_ydata(self.scenarios[self.current_drawing_scenario])
            self.fig.canvas.draw_idle()

    def predict_battery_times(self, name):
        charge = self.battery_charge_scenarios[name]
        load = self.scenarios[name]
        simulated_charge = charge
        t_full = None
        t_empty = None
        future_charges = []
        for t_future in range(self.time_steps):
            step_voltage = sum((self.temperature_sources[i][(self.t+t_future)%self.time_steps]+self.env_heat)*0.01
                               for i in range(self.num_sources))
            simulated_charge += step_voltage*self.efficiency - load[(self.t+t_future)%self.time_steps]
            simulated_charge = max(min(simulated_charge, self.battery_capacity),0)
            future_charges.append(simulated_charge)
            if simulated_charge >= self.battery_capacity and t_full is None:
                t_full = t_future
            if simulated_charge <= 0 and t_empty is None:
                t_empty = t_future
        return t_full, t_empty, future_charges

    def animate(self, frame):
        step_total_voltage = 0
        for i in range(self.num_sources):
            v = (self.temperature_sources[i][self.t]+self.env_heat)*0.01
            self.voltage_generated_sources[i][self.t] = v
            step_total_voltage += v

        for name, load_profile in self.scenarios.items():
            self.battery_charge_scenarios[name] += step_total_voltage*self.efficiency
            self.battery_charge_scenarios[name] -= load_profile[self.t]
            self.battery_charge_scenarios[name] = max(min(self.battery_charge_scenarios[name], self.battery_capacity),0)
            self.lines_charge[name].set_data(range(self.t+1), [self.battery_charge_scenarios[name]]*(self.t+1))

            # Update predictions
            t_full, t_empty, future_charges = self.predict_battery_times(name)
            text = ""
            if t_full is not None:
                text += f"Full in ~{t_full} steps; "
            else:
                text += "Won't reach full; "
            if t_empty is not None:
                text += f"Empty in ~{t_empty} steps"
            else:
                text += "Won't empty soon"
            self.prediction_texts[name].set_text(text)

            # Update timeline bar
            bar_width = len(future_charges)
            self.ax3.clear()
            self.ax3.barh(0, bar_width, left=0, color=self.colors[name], alpha=0.3)
            self.ax3.barh(0, future_charges[-1]/self.battery_capacity*bar_width, left=0, color=self.colors[name], alpha=0.8)
            self.ax3.axis('off')
            self.ax3.set_xlim(0, self.time_steps)

        self.heatmap.set_data(np.array(self.voltage_generated_sources)[:, :self.t+1])
        self.heatmap.set_extent([0,self.t,0,self.num_sources])

        self.t = (self.t+1)%self.time_steps
        return list(self.lines_charge.values()) + list(self.lines_load.values()) + list(self.prediction_texts.values()) + [self.heatmap]

# Launch simulator with visual timeline
ThermoChargeWithTimeline()
