(* math-qa case math-qa.test-functions.v1.
   Closed-form reference values for the optimisation test functions at
   fixed points. Guards against transcription drift between the
   mathematical definitions and the Rust implementations. *)

Off[General::munfl];
sphere[x_] := Total[x^2];
rosenbrock[x_] := Total[
  Table[100 (x[[i + 1]] - x[[i]]^2)^2 + (1 - x[[i]])^2, {i, Length[x] - 1}]
];
rastrigin[x_] := 10 Length[x] + Total[#^2 - 10 Cos[2 Pi #] & /@ x];
ackley[x_] := Module[{n = Length[x]},
  -20 Exp[-0.2 Sqrt[Total[x^2]/n]] - Exp[Total[Cos[2 Pi #] & /@ x]/n] + 20 + E
];

points = <|
  "sphere_123" -> {1., 2., 3.},
  "rosenbrock_11" -> {1., 1.},
  "rosenbrock_m11" -> {-1., 1.},
  "rastrigin_half" -> {0.5, -0.5},
  "ackley_00" -> {0., 0.},
  "ackley_11" -> {1., 1.}
|>;
payload = <|
  "schema_version" -> 1,
  "case" -> "math-qa.test-functions.v1",
  "values" -> <|
    "sphere_123" -> N[sphere[points["sphere_123"]], 16],
    "rosenbrock_11" -> N[rosenbrock[points["rosenbrock_11"]], 16],
    "rosenbrock_m11" -> N[rosenbrock[points["rosenbrock_m11"]], 16],
    "rastrigin_half" -> N[rastrigin[points["rastrigin_half"]], 16],
    "ackley_00" -> N[ackley[points["ackley_00"]], 16],
    "ackley_11" -> N[ackley[points["ackley_11"]], 16]
  |>
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
