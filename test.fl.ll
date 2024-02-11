; ModuleID = 'test.fl.bc'
source_filename = "test"

@0 = private unnamed_addr constant [14 x i8] c"Hello, World!\00", align 1

declare i8* @__const_string()

declare void @puts(i8*)

define i32 @main() {
_prelude:
  br label %body

body:                                             ; preds = %_prelude
  call void @puts(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @0, i32 0, i32 0))
  ret i32 123
}
