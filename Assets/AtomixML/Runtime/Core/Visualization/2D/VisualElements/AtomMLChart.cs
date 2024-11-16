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

        protected double x_min = 0;
        protected double x_max = 0;
        protected double y_min = 0;
        protected double y_max = 0;

        public Vector2 Plot(double x_normalized, double y_normalized)
        {
            var real_width = width - 2 * margin;
            var real_heigth = height - 2 * margin;
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

        }

        public void Refresh()
        {
            MarkDirtyRepaint();
        }
    }
}
