-- [CLI-100 BUILD-TARGET]
target("WeaselCliInstaller")
  set_kind("binary")
  add_files("*.cpp")
  add_rules("subcmd")
  add_syslinks("shell32", "advapi32")
