# HW5 Student Score Predictor
import wx
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from sklearn.linear_model import LinearRegression


class ScorePredictorFrame(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title="Student Score Predictor", size=(850, 650))

        data_path = Path(__file__).with_name("data.csv")
        self.data = pd.read_csv(data_path)

        X = self.data[["study_hours"]]
        y = self.data["score"]

        self.model = LinearRegression()
        self.model.fit(X, y)

        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        title = wx.StaticText(panel, label="Student Score Predictor")
        title_font = title.GetFont()
        title_font.PointSize += 8
        title_font = title_font.Bold()
        title.SetFont(title_font)

        info = wx.StaticText(
            panel,
            label="Enter study hours. The app predicts an exam score."
        )

        input_sizer = wx.BoxSizer(wx.HORIZONTAL)

        label = wx.StaticText(panel, label="Study Hours:")
        self.hours_input = wx.TextCtrl(panel, value="6")

        button = wx.Button(panel, label="Predict Score")
        button.Bind(wx.EVT_BUTTON, self.predict_score)

        input_sizer.Add(label, 0, wx.ALL | wx.CENTER, 5)
        input_sizer.Add(self.hours_input, 0, wx.ALL, 5)
        input_sizer.Add(button, 0, wx.ALL, 5)

        self.result_label = wx.StaticText(panel, label="Prediction will show here.")

        self.figure, self.ax = plt.subplots(figsize=(7, 4))
        self.canvas = FigureCanvas(panel, -1, self.figure)

        main_sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        main_sizer.Add(info, 0, wx.ALL | wx.CENTER, 5)
        main_sizer.Add(input_sizer, 0, wx.ALL | wx.CENTER, 5)
        main_sizer.Add(self.result_label, 0, wx.ALL | wx.CENTER, 10)
        main_sizer.Add(self.canvas, 1, wx.ALL | wx.EXPAND, 10)

        panel.SetSizer(main_sizer)

        self.draw_chart()
        self.Show()

    def draw_chart(self, predicted_hours=None, predicted_score=None):
        self.ax.clear()

        self.ax.scatter(
            self.data["study_hours"],
            self.data["score"],
            label="Real Data"
        )

        line_hours = pd.DataFrame({
            "study_hours": [
                self.data["study_hours"].min(),
                self.data["study_hours"].max()
            ]
        })

        line_scores = self.model.predict(line_hours)

        self.ax.plot(
            line_hours["study_hours"],
            line_scores,
            label="Prediction Line"
        )

        if predicted_hours is not None:
            self.ax.scatter(
                [predicted_hours],
                [predicted_score],
                marker="x",
                s=100,
                label="Your Prediction"
            )

        self.ax.set_title("Study Hours vs Exam Score")
        self.ax.set_xlabel("Study Hours")
        self.ax.set_ylabel("Exam Score")
        self.ax.legend()
        self.ax.grid(True)

        self.canvas.draw()

    def predict_score(self, event):
        try:
            hours = float(self.hours_input.GetValue())

            prediction = self.model.predict(
                pd.DataFrame({"study_hours": [hours]})
            )[0]

            self.result_label.SetLabel(
                f"Predicted exam score for {hours} study hours: {prediction:.1f}"
            )

            self.draw_chart(hours, prediction)

        except ValueError:
            wx.MessageBox(
                "Please enter a number.",
                "Error",
                wx.OK | wx.ICON_ERROR
            )


if __name__ == "__main__":
    app = wx.App()
    frame = ScorePredictorFrame()
    app.MainLoop()
