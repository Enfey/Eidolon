find_program(EIDOLON_LIT lit REQUIRED)
find_program(EIDOLON_FILECHECK FileCheck REQUIRED) # Tests find FileCheck via PATH, this isn't actually read but its 
                                                   # inclusion here prevents tests from looking like they've failed
                                                   # if FileCheck cannot be found

add_custom_target(eidolon-tests
  COMMAND ${EIDOLON_LIT} -v ${CMAKE_CURRENT_SOURCE_DIR}
          --param build_dir=$<TARGET_FILE_DIR:eidolon>   # Pass through to lit.cfg.py - build/tests
          --param tests_dir=${CMAKE_CURRENT_BINARY_DIR}  # Pass through to lit.cfg.py - build/bin
  DEPENDS eidolon
  USES_TERMINAL
  COMMENT "Running Eidolon tests..."
)