using UnityEngine;
using System;
using System.IO;
using UnityEditor;
using Sirenix.OdinInspector;

namespace Atomix.ML.Editor
{
    public class RandomTextureGenerator : MonoBehaviour
    {
        [SerializeField] private string _folderPath = "Assets/Editor/GeneratedTextures";


        [Button]
        public void GenerateTextures(int size, int count = 100, string baseName = "randomTexture")
        {
            // Create folder if it doesn't exist
            if (!Directory.Exists(_folderPath))
            {
                Directory.CreateDirectory(_folderPath);
                AssetDatabase.Refresh();
            }

            for (int i = 0; i < count; i++)
            {
                // Generate the texture
                Texture2D generatedTexture = GenerateRandomTexture(size);

                // Save the texture as a PNG
                SaveTextureAsPNG(generatedTexture, $"{_folderPath}/{baseName}_{size}x{size}_{i}.png");

                // Destroy the texture after saving to free memory
                //DestroyImmediate(generatedTexture);
            }

            AssetDatabase.Refresh();
            Debug.Log($"{count} textures generated and saved in {_folderPath}");
        }

        static Texture2D GenerateRandomTexture(int size)
        {
            Texture2D texture = new Texture2D(size, size, TextureFormat.RGB24, false);

            // Set each pixel to either black or white
            for (int x = 0; x < size; x++)
            {
                for (int y = 0; y < size; y++)
                {
                    // Randomly set the pixel color to white or black
                    Color color = UnityEngine.Random.value > 0.5f ? Color.white : Color.black;
                    texture.SetPixel(x, y, color);
                }
            }

            texture.Apply();
            return texture;
        }

        static void SaveTextureAsPNG(Texture2D texture, string filePath)
        {
            byte[] bytes = texture.EncodeToPNG();
            File.WriteAllBytes(filePath, bytes);
        }
    }

}

