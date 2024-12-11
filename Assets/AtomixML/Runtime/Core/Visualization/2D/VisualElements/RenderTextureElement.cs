using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;

namespace Atom.MachineLearning.Core.Visualization.VisualElements
{
    public class RenderTextureElement : VisualElement
    {
        public RenderTextureElement(Texture2D texture2D)
        {
            StyleBackground background = new StyleBackground(texture2D);
            this.style.backgroundImage = background;
        }
    }
}
