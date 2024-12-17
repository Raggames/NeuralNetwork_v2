using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;

namespace Atom.MachineLearning.Core.Visualization.VisualElements
{
    public abstract class AtomMLChart : VisualElement
    {
        protected int margin = 20; //px

        protected double width => style.width.value.value;
        protected double height => style.height.value.value;
        protected double real_width => width - 2 * margin;
        protected double real_heigth => height - 2 * margin;

        private Color _strokeColor = Color.black;
        public Color strokeColor { get { return _strokeColor; } set { _strokeColor = value; } }

        protected Color _backgroundColor = Color.white;
        public Color backgroundColor { get { return _backgroundColor; } set { _backgroundColor = value; style.backgroundColor = new StyleColor(_backgroundColor); } }

        protected double x_min = 0;
        protected double x_max = 0;
        protected double y_min = 0;
        protected double y_max = 0;

        public Vector2 Plot(double x_normalized, double y_normalized)
        {
            var x = (float)(margin + x_normalized * real_width);
            var y = (float)(margin + y_normalized * real_heigth);

            return new Vector2(x, y);
        }

        public void DrawGraduation()
        {
            // Implement graduation drawing logic here
            generateVisualContent += DrawOrthonormalLines;
            Refresh();
        }


        protected void DrawOrthonormalLines(MeshGenerationContext ctx)
        {
            var painter2D = ctx.painter2D;
            painter2D.BeginPath();

            painter2D.MoveTo(Plot(0, 1));
            painter2D.LineTo(Plot(1, 1));

            painter2D.MoveTo(Plot(0, 1));
            painter2D.LineTo(Plot(0, 0));

            painter2D.Stroke();
        }

        public void Refresh()
        {
            MarkDirtyRepaint();
        }
    }
}
