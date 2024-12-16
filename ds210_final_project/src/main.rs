use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use petgraph::graph::{Graph, NodeIndex};
use petgraph::algo::dijkstra;
use plotters::prelude::*;

#[derive(Debug)]
struct JobCategory {
    name: String,
    male_percentage: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let data = load_data("male-flight-attendants.tsv")?;
    let (graph, _node_indices) = create_graph(&data);

    let degrees = calculate_degrees(&graph);
    let two_hop_neighbors = calculate_two_hop_neighbors(&graph);

    analyze_distribution("Degree Distribution", &degrees);
    analyze_distribution("Two-Hop Neighbors Distribution", &two_hop_neighbors);

    plot_distribution("Degree Distribution", &degrees, "degree_distribution.png")?;
    plot_distribution("Two-Hop Neighbors Distribution", &two_hop_neighbors, "two_hop_distribution.png")?;

    Ok(())
}

fn load_data(filename: &str) -> Result<Vec<JobCategory>, Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();

    for (index, line) in reader.lines().enumerate() {
        if index == 0 { continue; } // Skip the header line
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let name = parts[..parts.len()-1].join(" ");
            if let Ok(male_percentage) = parts.last().unwrap().parse::<f64>() {
                data.push(JobCategory { name, male_percentage });
            } else {
                eprintln!("Warning: Could not parse male percentage for '{}'", name);
            }
        }
    }

    Ok(data)
}

fn create_graph(data: &[JobCategory]) -> (Graph<(String, f64), f64>, HashMap<String, NodeIndex>) {
    let mut graph = Graph::new();
    let mut node_indices = HashMap::new();

    for job in data {
        let node = graph.add_node((job.name.clone(), job.male_percentage));
        node_indices.insert(job.name.clone(), node);
    }

    for (i, job1) in data.iter().enumerate() {
        for job2 in data.iter().skip(i + 1) {
            let similarity = (job1.male_percentage - job2.male_percentage).abs();
            if similarity < 10.0 {  
                let node1 = node_indices[&job1.name];
                let node2 = node_indices[&job2.name];
                graph.add_edge(node1, node2, similarity);
            }
        }
    }

    println!("Graph created with {} nodes and {} edges.", graph.node_count(), graph.edge_count());
    (graph, node_indices)
}

fn calculate_degrees(graph: &Graph<(String, f64), f64>) -> Vec<usize> {
    graph.node_indices().map(|n| graph.neighbors(n).count()).collect()
}

fn calculate_two_hop_neighbors(graph: &Graph<(String, f64), f64>) -> Vec<usize> {
    graph.node_indices()
        .map(|n| {
            let distances = dijkstra(graph, n, None, |_| 1);
            distances.values().filter(|&&d| d == 2).count()
        })
        .collect()
}

fn analyze_distribution(name: &str, data: &[usize]) {
    if data.is_empty() {
        println!("No data available for {}", name);
        return;
    }

    let total: usize = data.iter().sum();
    let mean = total as f64 / data.len() as f64;
    
    if mean.is_nan() || mean.is_infinite() {
        println!("Mean calculation resulted in NaN or infinite value.");
        return;
    }

    let variance: f64 = data.iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum::<f64>() / data.len() as f64;
    
    let std_dev = variance.sqrt();

    println!("{} Analysis:", name);
    println!("  Mean: {:.2}", mean);
    println!("  Standard Deviation: {:.2}", std_dev);
    println!("  Minimum: {}", data.iter().min().unwrap());
    println!("  Maximum: {}", data.iter().max().unwrap());

    let log_data: Vec<f64> = data.iter().filter(|&&x| x > 0).map(|&x| (x as f64).ln()).collect();
    
    if !log_data.is_empty() {
        let (alpha, x_min) = estimate_power_law_parameters(&log_data);
        println!("  Estimated Power Law Parameters:");
        println!("    α: {:.2}", alpha);
        println!("    x_min: {:.2}", x_min.exp());

        let ks_statistic = kolmogorov_smirnov_test(&log_data, alpha, x_min);
        println!("  Kolmogorov-Smirnov Statistic: {:.4}", ks_statistic);
        
        if ks_statistic < 0.05 {
            println!("  The distribution closely follows a power-law (p < 0.05)");
        } else if ks_statistic < 0.1 {
            println!("  The distribution moderately follows a power-law (0.05 ≤ p < 0.1)");
        } else {
            println!("  The distribution does not strongly follow a power-law (p ≥ 0.1)");
        }
        
    } else {
        println!("Insufficient data for power law analysis");
    }
}

fn estimate_power_law_parameters(log_data: &[f64]) -> (f64, f64) {
    let n = log_data.len() as f64;
    
   if n <= 1.0 {
       return (f64::NAN, f64::NAN); 
   }
   
   let sum: f64 = log_data.iter().sum();
   let x_min = log_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
   let alpha = 1.0 + n / (sum - n * x_min.ln());
   (alpha, x_min)
}

fn kolmogorov_smirnov_test(log_data: &[f64], alpha: f64, x_min: f64) -> f64 {
   let sorted_data: Vec<f64> = log_data.iter().filter(|&&x| x >= x_min).cloned().collect();
   let m = sorted_data.len();

   sorted_data.iter().enumerate().map(|(i, &x)| {
       let theoretical_cdf = 1.0 - ((x / x_min).powf(-alpha + 1.0));
       let empirical_cdf = (i + 1) as f64 / m as f64;
       (theoretical_cdf - empirical_cdf).abs()
   }).fold(0.0, f64::max)
}

fn plot_distribution(title: &str, data: &[usize], filename: &str) -> Result<(), Box<dyn Error>> {
   let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
   root.fill(&WHITE)?;

   let max_value = *data.iter().max().unwrap_or(&1) as f64;
   let mut chart = ChartBuilder::on(&root)
       .caption(title, ("sans-serif", 40).into_font())
       .margin(5)
       .x_label_area_size(30)
       .y_label_area_size(30)
       .build_cartesian_2d((1.0..max_value).log_scale(), (1.0..data.len() as f64).log_scale())?;

   chart.configure_mesh().draw()?;

   chart.draw_series(
       data.iter().enumerate().map(|(i, &count)| {
           Circle::new((count as f64, (i + 1) as f64), 2, &RED.mix(0.5))
       })
   )?;

   chart.configure_series_labels()
       .background_style(&WHITE.mix(0.8))
       .border_style(&BLACK)
       .draw()?;

   root.present()?;
   Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_data() {
        let data = load_data("male-flight-attendants.tsv").unwrap();
        assert!(!data.is_empty());
        assert_eq!(data[0].name, "Kindergarten and earlier school teachers");
        assert_eq!(data[0].male_percentage, 2.3);
    }

    #[test]
    fn test_create_graph() {
        let data = vec![
            JobCategory { name: "Job1".to_string(), male_percentage: 50.0 },
            JobCategory { name: "Job2".to_string(), male_percentage: 60.0 },
            JobCategory { name: "Job3".to_string(), male_percentage: 70.0 },
        ];
        let (graph, node_indices) = create_graph(&data);
        assert_eq!(graph.node_count(), 3);
        assert!(graph.edge_count() > 0);
        assert_eq!(node_indices.len(), 3);
    }

    #[test]
    fn test_calculate_degrees() {
        let mut graph = Graph::new();
        let n1 = graph.add_node("1");
        let n2 = graph.add_node("2");
        let n3 = graph.add_node("3");
        graph.add_edge(n1, n2, ());
        graph.add_edge(n1, n3, ());
        let degrees = calculate_degrees(&graph);
        assert_eq!(degrees, vec![2, 1, 1]);
    }

    #[test]
    fn test_calculate_two_hop_neighbors() {
        let mut graph = Graph::new();
        let n1 = graph.add_node("1");
        let n2 = graph.add_node("2");
        let n3 = graph.add_node("3");
        let n4 = graph.add_node("4");
        graph.add_edge(n1, n2, ());
        graph.add_edge(n2, n3, ());
        graph.add_edge(n3, n4, ());
        let two_hop = calculate_two_hop_neighbors(&graph);
        assert_eq!(two_hop, vec![1, 1, 1, 1]);
    }
}
