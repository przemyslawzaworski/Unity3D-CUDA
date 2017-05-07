//copy HelloWorldCUDA.dll into Assets/Plugins directory
using System;
using UnityEngine;
using System.Runtime.InteropServices;

public class HelloWorldCUDA : MonoBehaviour 
{
	[DllImport("HelloWorldCUDA.dll", EntryPoint = "CUDA_device_name")]
	static extern IntPtr CUDA_device_name ();
	IntPtr handle;
	String caption;

	void Start()
	{
		handle = CUDA_device_name ();
		caption = Marshal.PtrToStringAnsi (handle);
	}

	void OnGUI() 
	{
		GUILayout.Label (caption);
	}
}
