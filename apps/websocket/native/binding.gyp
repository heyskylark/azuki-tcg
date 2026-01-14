{
  "targets": [
    {
      "target_name": "azuki_engine",
      "sources": [
        "src/addon.c"
      ],
      "include_dirs": [
        "../../../include",
        "../../../build/_deps/flecs_src-src/include"
      ],
      "libraries": [
        "-L<(module_root_dir)/../../../build",
        "-lazuki_lib",
        "-L<(module_root_dir)/../../../build/_deps/flecs_src-build",
        "-lflecs",
        "-lncurses"
      ],
      "conditions": [
        ["OS=='mac'", {
          "xcode_settings": {
            "MACOSX_DEPLOYMENT_TARGET": "10.15",
            "OTHER_LDFLAGS": [
              "-Wl,-rpath,@loader_path/../../../../../build",
              "-Wl,-rpath,@loader_path/../../../../../build/_deps/flecs_src-build"
            ]
          }
        }],
        ["OS=='linux'", {
          "ldflags": [
            "-Wl,-rpath,'$$ORIGIN/../../../../../build'",
            "-Wl,-rpath,'$$ORIGIN/../../../../../build/_deps/flecs_src-build'"
          ]
        }]
      ]
    }
  ]
}
