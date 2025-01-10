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
        public AxisChart(int width = 100, int height = 100)
        {
            style.width = new Length(width, LengthUnit.Percent);  // 50% of parent width
            style.height = new Length(height, LengthUnit.Percent);
            generateVisualContent += DrawOrthonormalLines_BottomLeftAnchored;
        }

        protected override void DrawOrthonormalLines_AutomaticCentered(MeshGenerationContext ctx)
        {
            throw new NotImplementedException();
        }
    }
}
