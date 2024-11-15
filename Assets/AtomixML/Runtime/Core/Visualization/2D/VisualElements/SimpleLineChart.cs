using Atom.MachineLearning.Core.Visualization.VisualElements;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;

namespace Atom.MachineLearning.Core.Visualization.VisualElements
{
    /// <summary>
    /// A simple graphic to input
    /// </summary>
    public class SimpleLineChart : AtomMLChart
    {
        /// <summary>
        /// Unidimensional mode, the points will be placed by the maximum avalaible interval on X axis
        /// If 500 px and 500 points, 1 point per pixel on X
        /// </summary>
        /// <param name="getPoints"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        public SimpleLineChart(Func<double[]> getPoints, int width = 300, int height = 300)
        {
            style.width = width;
            style.height = height;
            style.backgroundColor = new StyleColor(Color.white);

            generateVisualContent += GenerateVisualContent;
        }

        public SimpleLineChart(Func<double[,]> getPoints, int width = 300, int height = 300)
        {
            style.width = width;
            style.height = height;
            style.backgroundColor = new StyleColor(Color.white);

            generateVisualContent += GenerateVisualContent;
        }

        protected override void GenerateVisualContent(MeshGenerationContext ctx)
        {
            var painter2D = ctx.painter2D;

            painter2D.lineWidth = 5f;
            painter2D.strokeColor = Color.white;


        }
    }
}
