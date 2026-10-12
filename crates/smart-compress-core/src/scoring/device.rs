use crate::gpu::{GpuAdapterReport, adapter_report, select_adapters};

use super::{
    GpuWgpuBackend,
    shaders::{METRIC_SUMS_SHADER, TRANSFORM_DOWNSAMPLE_SHADER},
};

impl GpuWgpuBackend {
    pub fn new() -> anyhow::Result<Self> {
        Self::new_with_max_vram(None)
    }

    pub fn new_with_max_vram(max_vram_bytes: Option<u64>) -> anyhow::Result<Self> {
        Self::new_with_selection(max_vram_bytes, None, None, None)
    }

    pub fn new_with_selection(
        max_vram_bytes: Option<u64>,
        adapter_name: Option<&str>,
        adapter_vendor: Option<u32>,
        adapter_device: Option<u32>,
    ) -> anyhow::Result<Self> {
        let (adapter, _) = select_adapters(adapter_name, adapter_vendor, adapter_device)
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no wgpu adapter available"))?;
        Self::from_adapter(adapter, max_vram_bytes)
    }

    /// Creates the compute device from exactly this adapter.
    pub fn from_adapter(
        adapter: wgpu::Adapter,
        max_vram_bytes: Option<u64>,
    ) -> anyhow::Result<Self> {
        let adapter_report = adapter_report(&adapter);
        let adapter_limits = adapter.limits();
        let max_dispatch_bytes = [
            max_vram_bytes,
            Some(adapter_limits.max_buffer_size),
            Some(adapter_limits.max_storage_buffer_binding_size),
        ]
        .into_iter()
        .flatten()
        .min();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("smart-compress-gpu-metrics"),
                required_features: if adapter_report.timestamp_queries {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                },
                required_limits: adapter_limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))?;

        let metric_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smart-compress-metric-sums"),
            source: wgpu::ShaderSource::Wgsl(METRIC_SUMS_SHADER.into()),
        });
        let metric_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("smart-compress-metric-sums-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("smart-compress-metric-sums-pipeline-layout"),
            bind_group_layouts: &[Some(&metric_bind_group_layout)],
            immediate_size: 0,
        });
        let metric_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("smart-compress-metric-sums-pipeline"),
            layout: Some(&pipeline_layout),
            module: &metric_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let transform_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smart-compress-transform-downsample"),
            source: wgpu::ShaderSource::Wgsl(TRANSFORM_DOWNSAMPLE_SHADER.into()),
        });
        let transform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("smart-compress-transform-downsample-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let transform_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("smart-compress-transform-downsample-pipeline-layout"),
                bind_group_layouts: &[Some(&transform_bind_group_layout)],
                immediate_size: 0,
            });
        let transform_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("smart-compress-transform-downsample-pipeline"),
            layout: Some(&transform_pipeline_layout),
            module: &transform_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            metric_pipeline,
            metric_bind_group_layout,
            transform_pipeline,
            transform_bind_group_layout,
            max_dispatch_bytes,
            adapter_report,
        })
    }

    pub fn adapter_report(&self) -> &GpuAdapterReport {
        &self.adapter_report
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(super) fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
