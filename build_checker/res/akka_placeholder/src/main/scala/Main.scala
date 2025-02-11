// StdinStreamTyped.scala
// This example reads lines from standard input using a stream and prints each line.

import akka.actor.typed.scaladsl.Behaviors
import akka.actor.typed.{ActorSystem, Behavior}
import akka.stream.scaladsl.{Framing, Sink, StreamConverters}
import akka.util.ByteString
import akka.stream.SystemMaterializer
import scala.concurrent.ExecutionContext
import akka.{Done, NotUsed}
import scala.concurrent.Future

object StdinStreamApp extends App {
  // Create a simple root behavior that sets up the stream
  val root: Behavior[Nothing] = Behaviors.setup[Nothing] { context =>
    implicit val mat = SystemMaterializer(context.system).materializer
    implicit val ec: ExecutionContext = context.executionContext

    // Create a source from System.in
    val source: Source[String, NotUsed] = StreamConverters.fromInputStream(() => System.in)
      .via(
        Framing.delimiter(
          ByteString("\n"),         // Use ByteString for the newline delimiter
          maximumFrameLength = 256,
          allowTruncation = true
        )
      )
      .map(_.utf8String)             // Convert bytes to UTF-8 string

    // Sink that prints each line
    val sink: Sink[String, Future[Done]] = Sink.foreach(line => println(s"You typed: $line"))

    // Run the stream
    val done: Future[Done] = source.runWith(sink)
    done.onComplete { _ =>
      context.system.terminate()
    }

    Behaviors.empty
  }

  val system = ActorSystem(root, "StdinStreamSystem")
}
