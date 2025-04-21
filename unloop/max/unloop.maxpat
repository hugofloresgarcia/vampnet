{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 9,
			"minor" : 0,
			"revision" : 5,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 84.0, 131.0, 1000.0, 780.0 ],
		"gridsize" : [ 15.0, 15.0 ],
		"boxes" : [ 			{
				"box" : 				{
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"enablehscroll" : 0,
					"enablevscroll" : 0,
					"id" : "obj-2",
					"lockeddragscroll" : 0,
					"lockedsize" : 0,
					"maxclass" : "bpatcher",
					"name" : "unloop-bpatcher.maxpat",
					"numinlets" : 2,
					"numoutlets" : 2,
					"offset" : [ 0.0, 0.0 ],
					"outlettype" : [ "signal", "" ],
					"patching_rect" : [ 100.0, 68.0, 307.0, 469.0 ],
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 0,
					"patching_rect" : [ 143.0, 616.0, 55.0, 22.0 ],
					"text" : "dac~ 1 2"
				}

			}
 ],
		"lines" : [ 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 1 ],
					"order" : 0,
					"source" : [ "obj-2", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"order" : 1,
					"source" : [ "obj-2", 0 ]
				}

			}
 ],
		"originid" : "pat-142",
		"parameters" : 		{
			"obj-2::obj-1124" : [ "morph", "dry/wet", 0 ],
			"obj-2::obj-1125" : [ "level[8]", "level", 0 ],
			"obj-2::obj-1128" : [ "gain[4]", "gain", 0 ],
			"obj-2::obj-1140" : [ "overdub", "overdub", 0 ],
			"obj-2::obj-117" : [ "live.drop", "live.drop", 0 ],
			"obj-2::obj-1230" : [ "speed[2]", "speed+", 0 ],
			"obj-2::obj-171" : [ "toggle[2]", "toggle[30]", 0 ],
			"obj-2::obj-295" : [ "button[1]", "button[1]", 0 ],
			"obj-2::obj-316" : [ "toggle[3]", "toggle[3]", 0 ],
			"obj-2::obj-424::obj-12" : [ "number[8]", "number[2]", 0 ],
			"obj-2::obj-424::obj-13" : [ "number[9]", "number[3]", 0 ],
			"obj-2::obj-424::obj-15" : [ "number[2]", "number[2]", 0 ],
			"obj-2::obj-424::obj-19" : [ "number[3]", "number[3]", 0 ],
			"obj-2::obj-424::obj-20" : [ "number", "number", 0 ],
			"obj-2::obj-424::obj-23" : [ "number[4]", "number[3]", 0 ],
			"obj-2::obj-424::obj-26" : [ "number[5]", "number[3]", 0 ],
			"obj-2::obj-424::obj-28" : [ "number[6]", "number[2]", 0 ],
			"obj-2::obj-424::obj-30" : [ "number[7]", "number[2]", 0 ],
			"obj-2::obj-424::obj-347" : [ "periodic", "periodic", 0 ],
			"obj-2::obj-424::obj-349" : [ "drop", "drop", 0 ],
			"obj-2::obj-424::obj-8" : [ "toggle", "toggle", 0 ],
			"obj-2::obj-54" : [ "lpf", "lpf", 0 ],
			"obj-2::obj-55" : [ "tapelength", "length", 0 ],
			"obj-2::obj-76" : [ "hpf", "hpf", 0 ],
			"obj-2::obj-91::obj-156" : [ "live.gain~[26]", "live.gain~", 0 ],
			"obj-2::obj-91::obj-162" : [ "live.gain~[25]", "live.gain~", 0 ],
			"parameterbanks" : 			{
				"0" : 				{
					"index" : 0,
					"name" : "",
					"parameters" : [ "-", "-", "-", "-", "-", "-", "-", "-" ]
				}

			}
,
			"inherited_shortname" : 1
		}
,
		"dependency_cache" : [ 			{
				"name" : "dry-wet.maxpat",
				"bootpath" : "~/projects/research/unloop-2025/vampnet/unloop/max",
				"patcherrelativepath" : ".",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "unloop-bpatcher.maxpat",
				"bootpath" : "~/projects/research/unloop-2025/vampnet/unloop/max",
				"patcherrelativepath" : ".",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "vampnet-ui.maxpat",
				"bootpath" : "~/projects/research/unloop-2025/vampnet/unloop/max",
				"patcherrelativepath" : ".",
				"type" : "JSON",
				"implicit" : 1
			}
 ],
		"autosave" : 0
	}

}
