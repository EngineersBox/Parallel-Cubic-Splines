package com.engineersbox.pcsplines;

import org.apache.commons.lang3.ClassLoaderUtils;
import org.jocl.*;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

import static org.jocl.CL.*;

public class OpenCLParams {

    private final cl_context context;
    private final cl_command_queue queue;
    private final cl_program program;

    public OpenCLParams(final String filePath) {
        final long deviceType = CL_DEVICE_TYPE_ALL;
        CL.setExceptionsEnabled(true);

        final int[] platforms = new int[1];
        clGetPlatformIDs(0, null, platforms);
        final cl_platform_id[] platformIds = new cl_platform_id[platforms[0]];
        clGetPlatformIDs(
                platformIds.length,
                platformIds,
                null
        );
        final cl_platform_id platform = platformIds[0];

        final int[] devices = new int[1];
        clGetDeviceIDs(
                platform,
                deviceType,
                0,
                null,
                devices
        );
        final int deviceCount = devices[0];
        final cl_device_id[] deviceIds = new cl_device_id[deviceCount];
        clGetDeviceIDs(
                platform,
                deviceType,
                deviceCount,
                deviceIds,
                null
        );
        final cl_device_id device = deviceIds[0];

        final cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
        this.context = clCreateContext(
                contextProperties,
                1,
                new cl_device_id[]{device},
                null,
                null,
                null
        );

        final cl_queue_properties properties = new cl_queue_properties();
        this.queue = clCreateCommandQueueWithProperties(
                this.context,
                device,
                properties,
                null
        );

        final String programCode;
        try {
            programCode = Files.readString(Paths.get(Objects.requireNonNull(getClass().getResource(filePath)).toURI()));
        } catch (IOException | URISyntaxException e) {
            throw new RuntimeException(e);
        }
        this.program = clCreateProgramWithSource(
                this.context,
                1,
                new String[]{programCode},
                null,
                null
        );
        clBuildProgram(
                this.program,
                0,
                null,
                null,
                null,
                null
        );
    }

    public cl_context getContext() {
        return this.context;
    }

    public cl_command_queue getQueue() {
        return this.queue;
    }

    public cl_kernel getKernel(final String kernelName) {
        return clCreateKernel(this.program, kernelName, null);
    }

    public cl_program getProgram() {
        return this.program;
    }
}
