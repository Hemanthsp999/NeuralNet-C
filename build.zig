const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create a static library to collect all .o files
    const neuralNetCLib = b.addStaticLibrary(.{
        .name = "NeuralNet_C_Lib",
        .target = target,
        .optimize = optimize,
    });

    //const c_sources: [4][]const u8 = .{ "./EDA/DataAnalysis.c", "./Regression/Linear.c", "./Regression/memory_deallocation.c", "./Regression/model_performance_with_regularization.c" };
    const c_sources: [4][]const u8 = .{ "./File/file.c", "./File/memory.c", "./Activation/activation.c", "./Neural/neural.c" };

    neuralNetCLib.linkLibC();
    for (c_sources) |c_file| {
        neuralNetCLib.addCSourceFile(.{
            .file = b.path(c_file),
            .flags = &.{
                "-g",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I.",
                "-IRegression",
            },
        });
    }

    // Install the library (contains all .o files)
    b.installArtifact(neuralNetCLib);

    const test_exe = b.addExecutable(.{
        .name = "test",
        .target = target,
        .optimize = optimize,
    });

    // Link against the previously built library and libm
    test_exe.linkLibrary(neuralNetCLib);
    test_exe.linkSystemLibrary("m");

    test_exe.addCSourceFile(.{
        .file = b.path("./main.c"),
        .flags = &.{
            "-g",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I.",
            "-IRegression",
        },
    });

    // Make the test executable available
    b.installArtifact(test_exe);

    // Add a "test" step to run your test executable
    const test_step = b.step("test", "Run the tests");
    const run_test = b.addRunArtifact(test_exe);
    test_step.dependOn(&run_test.step);
}
