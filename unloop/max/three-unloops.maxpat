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
		"rect" : [ 34.0, 87.0, 1612.0, 929.0 ],
		"gridsize" : [ 15.0, 15.0 ],
		"boxes" : [ 			{
				"box" : 				{
					"id" : "obj-18",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1045.0, 739.0, 132.0, 22.0 ],
					"presentation_linecount" : 3,
					"text" : "text supermetroid"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-17",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 622.0, 739.0, 132.0, 22.0 ],
					"presentation_linecount" : 3,
					"text" : "text supermetroid"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-16",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 260.0, 739.0, 132.0, 22.0 ],
					"text" : "text supermetroid"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-12",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 678.0, 72.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-10",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 809.0, 107.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-6",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1137.0, 138.0, 150.0, 20.0 ],
					"text" : "infinite radio mode"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1025.0, 137.0, 110.0, 22.0 ],
					"presentation_linecount" : 2,
					"text" : "vampnet periodic 0"
				}

			}
, 			{
				"box" : 				{
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"enablehscroll" : 0,
					"enablevscroll" : 0,
					"id" : "obj-3",
					"lockeddragscroll" : 0,
					"lockedsize" : 0,
					"maxclass" : "bpatcher",
					"name" : "unloop-bpatcher.maxpat",
					"numinlets" : 2,
					"numoutlets" : 2,
					"offset" : [ 0.0, 0.0 ],
					"outlettype" : [ "signal", "" ],
					"patching_rect" : [ 876.0, 243.0, 301.0, 461.0 ],
					"viewvisibility" : 1
				}

			}
, 			{
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
					"patching_rect" : [ 463.0, 243.0, 301.0, 461.0 ],
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"enablehscroll" : 0,
					"enablevscroll" : 0,
					"id" : "obj-1",
					"lockeddragscroll" : 0,
					"lockedsize" : 0,
					"maxclass" : "bpatcher",
					"name" : "unloop-bpatcher.maxpat",
					"numinlets" : 2,
					"numoutlets" : 2,
					"offset" : [ 0.0, 0.0 ],
					"outlettype" : [ "signal", "" ],
					"patching_rect" : [ 91.0, 243.0, 301.0, 461.0 ],
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-33",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 478.0, 76.0, 150.0, 20.0 ],
					"text" : "play/stop the playhead"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-32",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 634.5, 35.0, 129.0, 33.0 ],
					"text" : "record into  all bufs simultaneously"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-31",
					"linecount" : 3,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 773.0, 45.0, 150.0, 47.0 ],
					"text" : "start generation \n(make sure client is running!) "
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-30",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 985.0, 103.0, 150.0, 20.0 ],
					"text" : "choose model option 14"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-28",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"patching_rect" : [ 916.0, 102.0, 58.0, 22.0 ],
					"text" : "loadbang"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-26",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 916.0, 137.0, 101.0, 22.0 ],
					"text" : "vampnet model 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-21",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 809.0, 143.0, 45.0, 22.0 ],
					"text" : "gen $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-15",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 678.0, 112.0, 42.0, 22.0 ],
					"text" : "rec $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-14",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patcher" : 					{
						"fileversion" : 1,
						"appversion" : 						{
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
						"boxes" : [ 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-2",
									"index" : 1,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 31.0, 145.0, 30.0, 30.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-1",
									"index" : 1,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 31.0, 20.0, 30.0, 30.0 ]
								}

							}
 ],
						"lines" : [ 							{
								"patchline" : 								{
									"destination" : [ "obj-2", 0 ],
									"source" : [ "obj-1", 0 ]
								}

							}
 ],
						"originid" : "pat-6"
					}
,
					"patching_rect" : [ 572.0, 184.0, 44.0, 22.0 ],
					"text" : "p pass"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-8",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 534.0, 112.0, 31.0, 22.0 ],
					"text" : "stop"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-7",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 500.0, 112.0, 31.0, 22.0 ],
					"text" : "play"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 0,
					"patching_rect" : [ 533.0, 750.0, 55.0, 22.0 ],
					"text" : "dac~ 1 2"
				}

			}
 ],
		"lines" : [ 			{
				"patchline" : 				{
					"destination" : [ "obj-16", 1 ],
					"source" : [ "obj-1", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5", 0 ],
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-21", 0 ],
					"source" : [ "obj-10", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-15", 0 ],
					"source" : [ "obj-12", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"order" : 2,
					"source" : [ "obj-14", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-2", 0 ],
					"order" : 1,
					"source" : [ "obj-14", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-3", 0 ],
					"order" : 0,
					"source" : [ "obj-14", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-14", 0 ],
					"source" : [ "obj-15", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-17", 1 ],
					"source" : [ "obj-2", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5", 1 ],
					"order" : 0,
					"source" : [ "obj-2", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5", 0 ],
					"order" : 1,
					"source" : [ "obj-2", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-14", 0 ],
					"source" : [ "obj-21", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-14", 0 ],
					"source" : [ "obj-26", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-26", 0 ],
					"order" : 1,
					"source" : [ "obj-28", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4", 0 ],
					"order" : 0,
					"source" : [ "obj-28", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-18", 1 ],
					"source" : [ "obj-3", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5", 1 ],
					"source" : [ "obj-3", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-14", 0 ],
					"source" : [ "obj-4", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-14", 0 ],
					"source" : [ "obj-7", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-14", 0 ],
					"source" : [ "obj-8", 0 ]
				}

			}
 ],
		"originid" : "pat-4",
		"parameters" : 		{
			"obj-1::obj-1124" : [ "morph", "dry/wet", 0 ],
			"obj-1::obj-1125" : [ "level[8]", "level", 0 ],
			"obj-1::obj-1128" : [ "gain[4]", "gain", 0 ],
			"obj-1::obj-1140" : [ "overdub", "overdub", 0 ],
			"obj-1::obj-117" : [ "live.drop", "live.drop", 0 ],
			"obj-1::obj-1230" : [ "speed[2]", "speed+", 0 ],
			"obj-1::obj-171" : [ "toggle[2]", "toggle[30]", 0 ],
			"obj-1::obj-295" : [ "button[1]", "button[1]", 0 ],
			"obj-1::obj-316" : [ "toggle[3]", "toggle[3]", 0 ],
			"obj-1::obj-424::obj-12" : [ "number[8]", "number[2]", 0 ],
			"obj-1::obj-424::obj-13" : [ "number[9]", "number[3]", 0 ],
			"obj-1::obj-424::obj-15" : [ "number[2]", "number[2]", 0 ],
			"obj-1::obj-424::obj-19" : [ "number[3]", "number[3]", 0 ],
			"obj-1::obj-424::obj-20" : [ "number", "number", 0 ],
			"obj-1::obj-424::obj-23" : [ "number[4]", "number[3]", 0 ],
			"obj-1::obj-424::obj-26" : [ "number[5]", "number[3]", 0 ],
			"obj-1::obj-424::obj-28" : [ "number[6]", "number[2]", 0 ],
			"obj-1::obj-424::obj-30" : [ "number[7]", "number[2]", 0 ],
			"obj-1::obj-424::obj-347" : [ "periodic", "periodic", 0 ],
			"obj-1::obj-424::obj-349" : [ "drop", "drop", 0 ],
			"obj-1::obj-424::obj-8" : [ "toggle", "toggle", 0 ],
			"obj-1::obj-54" : [ "lpf", "lpf", 0 ],
			"obj-1::obj-55" : [ "tapelength", "length", 0 ],
			"obj-1::obj-76" : [ "hpf", "hpf", 0 ],
			"obj-1::obj-91::obj-156" : [ "live.gain~[26]", "live.gain~", 0 ],
			"obj-1::obj-91::obj-162" : [ "live.gain~[25]", "live.gain~", 0 ],
			"obj-2::obj-1124" : [ "morph[1]", "dry/wet", 0 ],
			"obj-2::obj-1125" : [ "level[1]", "level", 0 ],
			"obj-2::obj-1128" : [ "gain[5]", "gain", 0 ],
			"obj-2::obj-1140" : [ "overdub[1]", "overdub", 0 ],
			"obj-2::obj-117" : [ "live.drop[1]", "live.drop", 0 ],
			"obj-2::obj-1230" : [ "speed[3]", "speed+", 0 ],
			"obj-2::obj-171" : [ "toggle[6]", "toggle[30]", 0 ],
			"obj-2::obj-295" : [ "button[2]", "button[1]", 0 ],
			"obj-2::obj-316" : [ "toggle[5]", "toggle[3]", 0 ],
			"obj-2::obj-424::obj-12" : [ "number[13]", "number[2]", 0 ],
			"obj-2::obj-424::obj-13" : [ "number[10]", "number[3]", 0 ],
			"obj-2::obj-424::obj-15" : [ "number[18]", "number[2]", 0 ],
			"obj-2::obj-424::obj-19" : [ "number[16]", "number[3]", 0 ],
			"obj-2::obj-424::obj-20" : [ "number[17]", "number", 0 ],
			"obj-2::obj-424::obj-23" : [ "number[15]", "number[3]", 0 ],
			"obj-2::obj-424::obj-26" : [ "number[12]", "number[3]", 0 ],
			"obj-2::obj-424::obj-28" : [ "number[14]", "number[2]", 0 ],
			"obj-2::obj-424::obj-30" : [ "number[11]", "number[2]", 0 ],
			"obj-2::obj-424::obj-347" : [ "periodic[1]", "periodic", 0 ],
			"obj-2::obj-424::obj-349" : [ "drop[1]", "drop", 0 ],
			"obj-2::obj-424::obj-8" : [ "toggle[4]", "toggle", 0 ],
			"obj-2::obj-54" : [ "lpf[1]", "lpf", 0 ],
			"obj-2::obj-55" : [ "tapelength[1]", "length", 0 ],
			"obj-2::obj-76" : [ "hpf[1]", "hpf", 0 ],
			"obj-2::obj-91::obj-156" : [ "live.gain~[2]", "live.gain~", 0 ],
			"obj-2::obj-91::obj-162" : [ "live.gain~[1]", "live.gain~", 0 ],
			"obj-3::obj-1124" : [ "morph[2]", "dry/wet", 0 ],
			"obj-3::obj-1125" : [ "level[2]", "level", 0 ],
			"obj-3::obj-1128" : [ "gain[6]", "gain", 0 ],
			"obj-3::obj-1140" : [ "overdub[2]", "overdub", 0 ],
			"obj-3::obj-117" : [ "live.drop[2]", "live.drop", 0 ],
			"obj-3::obj-1230" : [ "speed[4]", "speed+", 0 ],
			"obj-3::obj-171" : [ "toggle[9]", "toggle[30]", 0 ],
			"obj-3::obj-295" : [ "button[3]", "button[1]", 0 ],
			"obj-3::obj-316" : [ "toggle[8]", "toggle[3]", 0 ],
			"obj-3::obj-424::obj-12" : [ "number[26]", "number[2]", 0 ],
			"obj-3::obj-424::obj-13" : [ "number[24]", "number[3]", 0 ],
			"obj-3::obj-424::obj-15" : [ "number[21]", "number[2]", 0 ],
			"obj-3::obj-424::obj-19" : [ "number[19]", "number[3]", 0 ],
			"obj-3::obj-424::obj-20" : [ "number[25]", "number", 0 ],
			"obj-3::obj-424::obj-23" : [ "number[23]", "number[3]", 0 ],
			"obj-3::obj-424::obj-26" : [ "number[27]", "number[3]", 0 ],
			"obj-3::obj-424::obj-28" : [ "number[22]", "number[2]", 0 ],
			"obj-3::obj-424::obj-30" : [ "number[20]", "number[2]", 0 ],
			"obj-3::obj-424::obj-347" : [ "periodic[2]", "periodic", 0 ],
			"obj-3::obj-424::obj-349" : [ "drop[2]", "drop", 0 ],
			"obj-3::obj-424::obj-8" : [ "toggle[7]", "toggle", 0 ],
			"obj-3::obj-54" : [ "lpf[2]", "lpf", 0 ],
			"obj-3::obj-55" : [ "tapelength[2]", "length", 0 ],
			"obj-3::obj-76" : [ "hpf[2]", "hpf", 0 ],
			"obj-3::obj-91::obj-156" : [ "live.gain~[3]", "live.gain~", 0 ],
			"obj-3::obj-91::obj-162" : [ "live.gain~[4]", "live.gain~", 0 ],
			"parameterbanks" : 			{
				"0" : 				{
					"index" : 0,
					"name" : "",
					"parameters" : [ "-", "-", "-", "-", "-", "-", "-", "-" ]
				}

			}
,
			"parameter_overrides" : 			{
				"obj-2::obj-1124" : 				{
					"parameter_longname" : "morph[1]"
				}
,
				"obj-2::obj-1125" : 				{
					"parameter_longname" : "level[1]"
				}
,
				"obj-2::obj-1128" : 				{
					"parameter_longname" : "gain[5]"
				}
,
				"obj-2::obj-1140" : 				{
					"parameter_longname" : "overdub[1]"
				}
,
				"obj-2::obj-117" : 				{
					"parameter_longname" : "live.drop[1]"
				}
,
				"obj-2::obj-1230" : 				{
					"parameter_longname" : "speed[3]"
				}
,
				"obj-2::obj-424::obj-347" : 				{
					"parameter_longname" : "periodic[1]"
				}
,
				"obj-2::obj-424::obj-349" : 				{
					"parameter_longname" : "drop[1]"
				}
,
				"obj-2::obj-54" : 				{
					"parameter_longname" : "lpf[1]"
				}
,
				"obj-2::obj-55" : 				{
					"parameter_longname" : "tapelength[1]"
				}
,
				"obj-2::obj-76" : 				{
					"parameter_longname" : "hpf[1]"
				}
,
				"obj-2::obj-91::obj-156" : 				{
					"parameter_longname" : "live.gain~[2]"
				}
,
				"obj-2::obj-91::obj-162" : 				{
					"parameter_longname" : "live.gain~[1]"
				}
,
				"obj-3::obj-1124" : 				{
					"parameter_longname" : "morph[2]"
				}
,
				"obj-3::obj-1125" : 				{
					"parameter_longname" : "level[2]"
				}
,
				"obj-3::obj-1128" : 				{
					"parameter_longname" : "gain[6]"
				}
,
				"obj-3::obj-1140" : 				{
					"parameter_longname" : "overdub[2]"
				}
,
				"obj-3::obj-117" : 				{
					"parameter_longname" : "live.drop[2]"
				}
,
				"obj-3::obj-1230" : 				{
					"parameter_longname" : "speed[4]"
				}
,
				"obj-3::obj-424::obj-347" : 				{
					"parameter_longname" : "periodic[2]"
				}
,
				"obj-3::obj-424::obj-349" : 				{
					"parameter_longname" : "drop[2]"
				}
,
				"obj-3::obj-54" : 				{
					"parameter_longname" : "lpf[2]"
				}
,
				"obj-3::obj-55" : 				{
					"parameter_longname" : "tapelength[2]"
				}
,
				"obj-3::obj-76" : 				{
					"parameter_longname" : "hpf[2]"
				}
,
				"obj-3::obj-91::obj-156" : 				{
					"parameter_longname" : "live.gain~[3]"
				}
,
				"obj-3::obj-91::obj-162" : 				{
					"parameter_longname" : "live.gain~[4]"
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
