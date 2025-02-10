val scala3Version = "3.5.2"

lazy val root = project
  .in(file("."))
  .settings(
    name := "akka_placeholder",
    version := "0.1.0-SNAPSHOT",
    scalaVersion := scala3Version,
    libraryDependencies ++= Seq(
      // Testing library
      "org.scalameta" %% "munit" % "1.0.0" % Test,
      
      // Akka Typed API
      "com.typesafe.akka" %% "akka-actor-typed" % "2.8.2",
      
      // Akka Cluster (Typed and Classic)
      "com.typesafe.akka" %% "akka-cluster-typed" % "2.8.2",
      "com.typesafe.akka" %% "akka-cluster" % "2.8.2",
      
      // Akka Persistence (Typed and Classic)
      "com.typesafe.akka" %% "akka-persistence-typed" % "2.8.2",
      "com.typesafe.akka" %% "akka-persistence" % "2.8.2",
      
      // Akka Streams for stream processing integration
      "com.typesafe.akka" %% "akka-stream" % "2.8.2",
      
      // Akka Actor Classic (useful for context.become and other classic APIs)\n      "com.typesafe.akka" %% "akka-actor" % "2.8.2",
      
      // SLF4J logging support for Akka\n      "com.typesafe.akka" %% "akka-slf4j" % "2.8.2",
      
      // Typesafe Config for configuration management\n      "com.typesafe" % "config" % "1.4.2"
    )
  )
