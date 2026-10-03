file(REMOVE_RECURSE
  "libazuki_lib.a"
  "libazuki_lib.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/azuki_lib.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
