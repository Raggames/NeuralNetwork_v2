using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;

namespace Atom.MachineLearning.Core.Visualization.VisualElements
{
    public class AxisChart : AtomMLChart
    {
        public AxisChart(int width = 300, int height = 300)
        {
            style.width = width;
            style.height = height;
            style.left = 0;   // Distance from the left edge
            style.bottom = 0; // Distance from the bottom edge
            generateVisualContent += DrawOrthonormalLines;
        }
    }
}
